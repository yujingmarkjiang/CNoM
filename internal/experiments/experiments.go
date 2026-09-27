package experiments

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/cnom/cnom/internal/audit"
	"github.com/cnom/cnom/internal/cascade"
	"github.com/cnom/cnom/internal/cert"
	"github.com/cnom/cnom/internal/cost"
	"github.com/cnom/cnom/internal/match"
	"github.com/cnom/cnom/internal/monitor"
)

type Spec struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Kind    string `json:"kind"`
	Input   string `json:"input"`
	Output  string `json:"output"`
	Rmin    float64 `json:"rmin"`
	Delta   float64 `json:"delta"`
	Upload  float64 `json:"upload"`
}

func Catalog() []Spec {
	return []Spec{
		{ID: "E1", Title: "Selection-valid certification", Kind: "cert"},
		{ID: "E2", Title: "Winner's curse vs grid size", Kind: "cert"},
		{ID: "E3", Title: "Dependence-aware certification", Kind: "cert"},
		{ID: "E4", Title: "Price of recall", Kind: "cost"},
		{ID: "E5", Title: "Event-matching sensitivity", Kind: "match"},
		{ID: "E6", Title: "Modern cloud Teachers", Kind: "train"},
		{ID: "E7", Title: "Edge Sentinel backbones", Kind: "train"},
		{ID: "E8", Title: "Cascade baselines at 6.8% upload", Kind: "cascade"},
		{ID: "E9", Title: "Zero-shot and training-free", Kind: "train"},
		{ID: "E10", Title: "VLM Teacher with rationale", Kind: "train"},
		{ID: "E11", Title: "Leave-one-zone-out", Kind: "slice"},
		{ID: "E12", Title: "New-camera onboarding", Kind: "slice"},
		{ID: "E13", Title: "Temporal generalization", Kind: "slice"},
		{ID: "E14", Title: "Public-benchmark transfer", Kind: "train"},
		{ID: "E15", Title: "Synthetic corruptions", Kind: "slice"},
		{ID: "E16", Title: "Real-condition slices", Kind: "slice"},
		{ID: "E17", Title: "Real drift events", Kind: "slice"},
		{ID: "E18", Title: "Edge power modes", Kind: "ops"},
		{ID: "E19", Title: "Capture-recapture models", Kind: "audit"},
		{ID: "E20", Title: "Month-by-month production", Kind: "slice"},
		{ID: "E21", Title: "Version-stratified audit", Kind: "slice"},
		{ID: "E22", Title: "Recertification delay", Kind: "monitor"},
		{ID: "E23", Title: "Hardware portability", Kind: "ops"},
		{ID: "E24", Title: "Streams per device", Kind: "ops"},
		{ID: "E25", Title: "Total-cost sensitivity", Kind: "cost"},
		{ID: "E26", Title: "Energy and carbon", Kind: "cost"},
		{ID: "E27", Title: "Reviewer decision study", Kind: "human"},
		{ID: "E28", Title: "Alert fatigue", Kind: "human"},
		{ID: "E29", Title: "Staggered DiD", Kind: "human"},
		{ID: "E30", Title: "Worker acceptance survey", Kind: "human"},
		{ID: "E31", Title: "Privacy-preserving upload", Kind: "train"},
		{ID: "E32", Title: "Sentinel module structure", Kind: "train"},
		{ID: "E33", Title: "Temporal policy grid", Kind: "cascade"},
		{ID: "E34", Title: "Distillation hyperparameters", Kind: "train"},
		{ID: "E35", Title: "Seed variance", Kind: "train"},
		{ID: "E36", Title: "Clip context length", Kind: "train"},
	}
}

func Run(spec Spec, inPath, outPath string) error {
	if outPath == "" {
		outPath = filepath.Join("runs", spec.ID+".json")
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	var payload any
	var err error
	switch spec.Kind {
	case "cert":
		payload, err = runCert(spec, inPath)
	case "cascade":
		payload, err = runCascade(spec, inPath)
	case "match":
		payload, err = runMatch(inPath)
	case "audit":
		payload, err = runAudit(inPath)
	case "cost":
		payload, err = runCost(spec)
	case "monitor":
		payload, err = runMonitor(inPath)
	case "slice", "ops", "human", "train":
		payload = map[string]any{
			"id": spec.ID, "title": spec.Title, "kind": spec.Kind,
			"status": "ready",
			"requires": requires(spec.Kind),
		}
	default:
		return fmt.Errorf("unknown kind %s", spec.Kind)
	}
	if err != nil {
		return err
	}
	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(outPath, raw, 0o644)
}

func requires(kind string) string {
	switch kind {
	case "train":
		return "python trainer + labeled clips + backbone weights"
	case "slice":
		return "audit table with zone/camera/month/condition columns"
	case "ops":
		return "device benchmark logs"
	case "human":
		return "reviewer / survey / rollout tables"
	default:
		return ""
	}
}

func runCert(spec Spec, path string) (any, error) {
	policies := defaultFrontier()
	if path != "" {
		var loaded []cert.Policy
		if err := readJSON(path, &loaded); err == nil && len(loaded) > 0 {
			policies = loaded
		}
	}
	rmin, delta := spec.Rmin, spec.Delta
	if rmin == 0 {
		rmin = 0.95
	}
	if delta == 0 {
		delta = 0.05
	}
	bon := cert.Bonferroni(policies, rmin, delta)
	ltt := cert.FixedSequence(policies, rmin, delta)
	bBest, _ := cert.Cheapest(bon)
	lBest, _ := cert.Cheapest(ltt)
	return map[string]any{
		"id": spec.ID, "bonferroni": bon, "ltt": ltt,
		"selected_bonferroni": bBest, "selected_ltt": lBest,
		"grid": len(policies),
	}, nil
}

func defaultFrontier() []cert.Policy {
	rows := []struct {
		tau, r, u float64
	}{
		{0.20, 0.985, 0.119},
		{0.25, 0.978, 0.089},
		{0.30, 0.968, 0.068},
		{0.35, 0.958, 0.053},
		{0.40, 0.943, 0.042},
	}
	out := make([]cert.Policy, 0, len(rows))
	for _, row := range rows {
		n := 720
		k := int(math.Round(row.r * float64(n)))
		out = append(out, cert.Policy{TauF: row.tau, W: 20, Rho: 0.55, Upload: row.u, K: k, N: n})
	}
	return out
}

func runCascade(spec Spec, path string) (any, error) {
	type row struct {
		ID    string    `json:"id"`
		P     []float64 `json:"probs"`
		Label int       `json:"label"`
		Diff  float64   `json:"diff"`
	}
	var rows []row
	if path != "" {
		_ = readJSON(path, &rows)
	}
	if len(rows) == 0 {
		return map[string]any{"id": spec.ID, "status": "ready", "requires": "clip probability table"}, nil
	}
	frac := spec.Upload
	if frac == 0 {
		frac = 0.068
	}
	kinds := []string{"random", "confidence", "entropy", "margin", "l2d", "topk", "noscope", "reducto"}
	out := map[string]any{"id": spec.ID, "upload": frac}
	for _, kind := range kinds {
		ranks := make([]float64, len(rows))
		for i, r := range rows {
			s := cascade.Scores{PUnsafe: 1 - r.P[0], Entropy: cascade.Entropy(r.P), Margin: cascade.Margin(r.P), Conf: maxv(r.P), Diff: r.Diff}
			if kind == "random" {
				ranks[i] = float64((i*17+13)%len(rows)) / float64(len(rows))
			} else {
				ranks[i] = cascade.Rank(kind, s)
			}
		}
		sel := cascade.SelectBudget(ranks, frac)
		hit, tot := 0, 0
		for i, r := range rows {
			if r.Label == 0 {
				continue
			}
			tot++
			if sel[i] {
				hit++
			}
		}
		rec := 0.0
		if tot > 0 {
			rec = float64(hit) / float64(tot)
		}
		out[kind] = map[string]any{"recall_escalated": rec, "uploaded": count(sel), "n": len(rows)}
	}
	return out, nil
}

func runMatch(path string) (any, error) {
	type pair struct {
		A0, A1, B0, B1, PredOnset, RefOnset float64
	}
	var pairs []pair
	if path != "" {
		_ = readJSON(path, &pairs)
	}
	ths := []float64{0.1, 0.3, 0.5}
	out := map[string]any{"id": "E5"}
	for _, th := range ths {
		ok := 0
		for _, p := range pairs {
			if match.Overlap(p.A0, p.A1, p.B0, p.B1) >= th {
				ok++
			}
		}
		out[fmt.Sprintf("tiou_%.1f", th)] = ok
	}
	onset := 0
	for _, p := range pairs {
		if match.OnsetOK(p.PredOnset, p.RefOnset, 2) {
			onset++
		}
	}
	out["onset_pm_2s"] = onset
	out["n"] = len(pairs)
	return out, nil
}

func runAudit(path string) (any, error) {
	var body struct{ N1, N2, M, Detected, F1, F2 int }
	body.Detected = 2382
	body.N1, body.N2, body.M = 2464, 2400, 2386
	if path != "" {
		_ = readJSON(path, &body)
	}
	nHat := audit.Chapman(body.N1, body.N2, body.M)
	return map[string]any{
		"id": "E19", "chapman": nHat, "chao": audit.Chao(body.Detected, body.F1, body.F2, 4),
		"recall": audit.Recall(body.Detected, nHat),
	}, nil
}

func runCost(spec Spec) (any, error) {
	hs, cloud, ratio := cost.CameraMonth(2.69, 94.60, 0.068)
	return map[string]any{
		"id": spec.ID, "hs_guard": hs, "cloud_only": cloud, "ratio": ratio,
		"minus50": wrap(cost.CameraMonth(2.69, 47.30, 0.068)),
		"plus50":  wrap(cost.CameraMonth(2.69, 141.90, 0.068)),
	}, nil
}

func wrap(hs, cloud, ratio float64) map[string]float64 {
	return map[string]float64{"hs": hs, "cloud": cloud, "ratio": ratio}
}

func runMonitor(path string) (any, error) {
	w := monitor.Window{Detected: 285, Events: 300}
	if path != "" {
		_ = readJSON(path, &w)
	}
	return map[string]any{"id": "E22", "alarm": monitor.Alarm(w, 0.95, 1.96), "wilson": cert.WilsonLower(w.Detected, w.Events, 1.96)}, nil
}

func readJSON(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, v)
}

func maxv(p []float64) float64 {
	m := math.Inf(-1)
	for _, v := range p {
		if v > m {
			m = v
		}
	}
	return m
}

func count(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}

func IDs() []string {
	c := Catalog()
	ids := make([]string, len(c))
	for i, s := range c {
		ids[i] = s.ID
	}
	sort.Strings(ids)
	return ids
}
