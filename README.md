# CNoM

C++ edge runtime, Go cloud/certification, Python training. Policy: `τ_f=0.30`, `W=20`, `ρ=0.55`, merge 3 s, cooldown 8 s, `Δ=1.6 s`, `τ_u=0.80`, `τ_s=0.985`, `T_cal=1.34`.

```bash
make test
pip install -e ".[train]"
cnom-edge scores.tsv http://127.0.0.1:8080
CNOM_API_KEY=... cnomd -addr :8080
cnom-exp list
cnom train configs/experiments/teacher_vivit.yaml
cnom exp E1 configs/default.yaml
```
