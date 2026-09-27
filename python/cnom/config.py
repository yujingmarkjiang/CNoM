from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?}")


class ConfigNode(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def clone(self) -> "ConfigNode":
        return to_node(copy.deepcopy(dict(self)))


def to_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({k: to_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_node(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"Environment variable {name} is required")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _read_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Recursive config include: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    defaults = data.pop("defaults", [])
    merged: dict[str, Any] = {}
    for item in defaults:
        include = Path(item)
        if not include.is_absolute():
            include = path.parent / include
        merged = deep_merge(merged, _read_yaml(include, seen))
    seen.remove(path)
    return deep_merge(merged, data)


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_dotpath(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: Any = data
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if isinstance(cursor, list):
            if not part.isdigit():
                raise ValueError(f"List override segment must be an index: {part}")
            position = int(part)
            while len(cursor) <= position:
                cursor.append(None)
            if last:
                cursor[position] = value
                return
            expect_list = parts[index + 1].isdigit()
            if not isinstance(cursor[position], list if expect_list else dict):
                cursor[position] = [] if expect_list else {}
            cursor = cursor[position]
            continue
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot traverse override path at {part}")
        if last:
            cursor[part] = value
            return
        expect_list = parts[index + 1].isdigit()
        expected_type = list if expect_list else dict
        if not isinstance(cursor.get(part), expected_type):
            cursor[part] = [] if expect_list else {}
        cursor = cursor[part]


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ConfigNode:
    config_path = Path(path).resolve()
    data = _read_yaml(config_path, set())
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value: {override}")
        key, value = override.split("=", 1)
        set_dotpath(data, key.strip(), _parse_value(value.strip()))
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    return to_node(_expand_env(data))


def save_config(config: ConfigNode | dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json.loads(json.dumps(config)), handle, sort_keys=False, allow_unicode=True)


def select(config: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
