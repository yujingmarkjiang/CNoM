from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from cnom.config import load_config
from cnom.utils import configure_logging, finalize_distributed, read_table

app = typer.Typer(name="cnom", no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()
Override = Annotated[list[str] | None, typer.Option("--set", help="Override config values with key=value; repeat as needed")]


def _config(path: str, overrides: list[str] | None):
    return load_config(path, overrides or ())


def _show(value) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict(orient="records")
    console.print_json(json.dumps(value, default=str))


@app.command("prepare-manifest")
def prepare_manifest_command(config: str, input_root: str | None = None, output: str | None = None, set_values: Override = None) -> None:
    from cnom.data.adapters import prepare_manifest
    frame = prepare_manifest(_config(config, set_values), input_root=input_root, output=output)
    _show({"samples": len(frame), "output": output or _config(config, set_values).get("output")})


@app.command("audit-manifest")
def audit_manifest_command(manifest: str, root: str | None = None, strict_paths: bool = True, grouped_fields: str = "source_id,camera_id,scene_id,cluster_id") -> None:
    from cnom.validation import validate_manifest_file
    _show(validate_manifest_file(manifest, root, tuple(value.strip() for value in grouped_fields.split(",") if value.strip()), strict_paths))


@app.command("extract-features")
def extract_features_command(config: str, split: str | None = None, set_values: Override = None) -> None:
    from cnom.features import extract_feature_bags
    frame = extract_feature_bags(_config(config, set_values), split)
    _show({"samples": len(frame), "with_features": int(frame["feature_path"].notna().sum())})


@app.command("precompute-grounding")
def grounding_command(config: str, split: str | None = None, set_values: Override = None) -> None:
    from cnom.grounding import precompute_grounding
    frame = precompute_grounding(_config(config, set_values), split)
    _show({"samples": len(frame), "with_grounding": int(frame["object_features_path"].notna().sum())})


@app.command("mine-hard-negatives")
def mine_hard_negatives_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.hard_negatives import mine_hard_negatives
    frame = mine_hard_negatives(_config(config, set_values))
    _show({"candidates": len(frame), "mean_score": float(frame["mining_score"].mean()) if len(frame) else 0.0})


@app.command("build-hard-negative-replay")
def build_hard_negative_replay_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.hard_negatives import build_hard_negative_replay
    frame = build_hard_negative_replay(_config(config, set_values))
    _show({"accepted_hard_negatives": len(frame)})


@app.command("fingerprint-manifest")
def fingerprint_manifest_command(config: str, set_values: Override = None) -> None:
    from cnom.data.fingerprints import fingerprint_manifest
    outputs = fingerprint_manifest(_config(config, set_values))
    _show({name: len(frame) for name, frame in outputs.items()})


@app.command("dataset-report")
def dataset_report_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.dataset_report import build_dataset_report
    outputs = build_dataset_report(_config(config, set_values))
    _show({name: len(frame) for name, frame in outputs.items()})


@app.command("train")
def train_command(config: str, set_values: Override = None, log_level: str = "INFO", json_logs: bool = False) -> None:
    from cnom.runner import train_experiment
    configure_logging(log_level, json_logs)
    try:
        _show(train_experiment(_config(config, set_values)))
    finally:
        finalize_distributed()


@app.command("evaluate")
def evaluate_command(config: str, split: str = "test", set_values: Override = None) -> None:
    from cnom.runner import evaluate_experiment
    _, metrics = evaluate_experiment(_config(config, set_values), split)
    _show(metrics)


@app.command("fit-temperature")
def temperature_command(config: str, logits: str, output: str, set_values: Override = None) -> None:
    from cnom.runner import fit_temperature
    _show({"temperature": fit_temperature(_config(config, set_values), logits, output), "output": output})


@app.command("cache-frame-scores")
def cache_scores_command(config: str, output: str, split: str = "validation", checkpoint: str | None = None, set_values: Override = None) -> None:
    from cnom.inference import cache_frame_scores
    from cnom.models.registry import build_model
    cfg = _config(config, set_values)
    model_cfg = dict(cfg.get("sentinel", cfg.get("model", {})))
    model = build_model(model_cfg, checkpoint=checkpoint or model_cfg.get("checkpoint"), strict=False)
    frame = cache_frame_scores(cfg, model, split, output)
    _show({"frames": len(frame), "output": output})


@app.command("calibrate")
def calibrate_command(config: str, frame_scores: str, output_dir: str | None = None, set_values: Override = None) -> None:
    from cnom.analysis.calibration import calibrate_policy
    from cnom.inference import CandidateTeacherScorer
    from cnom.metrics.events import events_from_frame
    from cnom.models.registry import build_model
    from cnom.data.manifest import load_manifest, load_stream_manifest
    cfg = _config(config, set_values)
    scores = read_table(frame_scores)
    dataset_cfg = cfg["dataset"]
    events = load_manifest(dataset_cfg["events"], root=dataset_cfg.get("root"), strict_paths=False)
    teacher_cfg = dict(cfg.get("teacher", cfg.get("model", {})))
    teacher = build_model(teacher_cfg, checkpoint=teacher_cfg.get("checkpoint"), strict=False)
    source_map = dict(scores[["source_id", "video_path"]].drop_duplicates().astype(str).values)
    scorer = CandidateTeacherScorer(teacher, source_map, cfg)
    split = str(dataset_cfg.get("validation_split", "validation"))
    streams = load_stream_manifest(dataset_cfg, strict_paths=False)
    streams = streams.loc[streams["split"].astype(str).eq(split)]
    duration = float((streams["end_s"] - streams["start_s"]).sum())
    truth = events_from_frame(events.loc[events["split"].astype(str).eq(split)] if "split" in events.columns else events)
    selected, _ = calibrate_policy(scores, truth, scorer, cfg, duration / 3600, duration, output_dir or str(Path(cfg["output_dir"]) / "calibration"))
    _show(selected)


@app.command("benchmark")
def benchmark_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.benchmarks import run_benchmark_matrix
    _show(run_benchmark_matrix(_config(config, set_values)))


@app.command("ablate")
def ablation_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.ablations import run_ablation_matrix
    try:
        _show(run_ablation_matrix(_config(config, set_values)))
    finally:
        finalize_distributed()


@app.command("low-label")
def low_label_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.low_label import run_low_label_study
    _show(run_low_label_study(_config(config, set_values)))


@app.command("external")
def external_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.external import run_external_benchmarks
    _show(run_external_benchmarks(_config(config, set_values)))


@app.command("vlm-diagnostic")
def vlm_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.vlm import run_vlm_diagnostic
    _show(run_vlm_diagnostic(_config(config, set_values)))


@app.command("network-stress")
def network_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.network import run_network_stress
    _show(asyncio.run(run_network_stress(_config(config, set_values))))


@app.command("codec-study")
def codec_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.codec import run_codec_study
    _show(run_codec_study(_config(config, set_values)))


@app.command("resource-analysis")
def resource_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.resources import analyze_resource_usage
    _show(analyze_resource_usage(_config(config, set_values)))


@app.command("capacity-study")
def capacity_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.resources import run_capacity_study
    _show(run_capacity_study(_config(config, set_values)))


@app.command("latency-analysis")
def latency_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.latency import analyze_latency
    outputs = analyze_latency(_config(config, set_values))
    _show({name: len(frame) for name, frame in outputs.items()})


@app.command("deployment-analysis")
def deployment_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.deployment import analyze_deployment
    _show(analyze_deployment(_config(config, set_values)))


@app.command("evidence-bundle")
def evidence_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.evidence import build_evidence_bundle
    _show(build_evidence_bundle(_config(config, set_values)))


@app.command("operational-analysis")
def operational_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.operational import analyze_operational_outcomes
    _show(analyze_operational_outcomes(_config(config, set_values)))


@app.command("failure-analysis")
def failure_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.failures import analyze_failures
    outputs = analyze_failures(_config(config, set_values))
    _show({name: len(frame) for name, frame in outputs.items()})


@app.command("drift-analysis")
def drift_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.drift import analyze_drift
    _show(analyze_drift(_config(config, set_values)))


@app.command("release-check")
def release_command(config: str, set_values: Override = None) -> None:
    from cnom.analysis.release import check_release
    _show(check_release(_config(config, set_values)))


@app.command("exp")
def exp_command(experiment: str, config: str | None = None, set_values: Override = None) -> None:
    from cnom.experiments.run import EXPERIMENTS, run
    cfg = _config(config, set_values) if config else {"output_dir": "runs"}
    if experiment in {"list", "all"}:
        ids = list(EXPERIMENTS)
        if experiment == "list":
            _show(ids)
            return
        _show({eid: run(eid, cfg) for eid in ids})
        return
    _show(run(experiment, cfg))


@app.command("export")
def export_command(config: str, target: str = "torchscript", output: str | None = None, set_values: Override = None) -> None:
    from cnom.export import export_model
    _show({"output": str(export_model(_config(config, set_values), target, output))})


@app.command("edge")
def edge_command(config: str, set_values: Override = None, log_level: str = "INFO") -> None:
    from cnom.pipeline.edge import EdgeAgent
    configure_logging(log_level)
    EdgeAgent(_config(config, set_values)).run()


@app.command("cloud-api")
def cloud_api_command(config: str, set_values: Override = None) -> None:
    from cnom.service.api import run_api
    run_api(_config(config, set_values))


@app.command("cloud-worker")
def cloud_worker_command(config: str, set_values: Override = None, log_level: str = "INFO") -> None:
    from cnom.service.worker import CloudWorker
    configure_logging(log_level)
    CloudWorker(_config(config, set_values)).run()


@app.command("validate-results")
def validate_results_command(path: str) -> None:
    from cnom.validation import validate_result_directory
    _show(validate_result_directory(path))


@app.command("config")
def show_config_command(config: str, set_values: Override = None) -> None:
    _show(_config(config, set_values))


if __name__ == "__main__":
    app()
