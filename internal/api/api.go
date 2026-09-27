package api

import (
	"encoding/json"
	"net/http"
	"sync"

	"github.com/cnom/cnom/internal/audit"
	"github.com/cnom/cnom/internal/cert"
	"github.com/cnom/cnom/internal/monitor"
	"github.com/cnom/cnom/internal/router"
	"github.com/cnom/cnom/internal/types"
)

type Server struct {
	Key    string
	Router router.Router
	mu     sync.Mutex
	jobs   []types.ClipJob
	audit  monitor.Window
}

func New(key string) *Server {
	return &Server{Key: key, Router: router.New()}
}

func (s *Server) auth(w http.ResponseWriter, r *http.Request) bool {
	if s.Key == "" {
		return true
	}
	if r.Header.Get("Authorization") != "Bearer "+s.Key {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"ok":true}`))
	})
	mux.HandleFunc("POST /v1/clips", s.ingest)
	mux.HandleFunc("POST /v1/route", s.route)
	mux.HandleFunc("POST /v1/certify", s.certify)
	mux.HandleFunc("POST /v1/capture-recapture", s.cr)
	mux.HandleFunc("POST /v1/audit-event", s.auditEvent)
	mux.HandleFunc("GET /v1/monitor", s.mon)
	return mux
}

func (s *Server) ingest(w http.ResponseWriter, r *http.Request) {
	if !s.auth(w, r) {
		return
	}
	var job types.ClipJob
	if err := json.NewDecoder(r.Body).Decode(&job); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	s.mu.Lock()
	s.jobs = append(s.jobs, job)
	s.mu.Unlock()
	writeJSON(w, map[string]any{"accepted": true, "id": job.ID})
}

func (s *Server) route(w http.ResponseWriter, r *http.Request) {
	if !s.auth(w, r) {
		return
	}
	var body struct {
		Probs []float64 `json:"probs"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	writeJSON(w, s.Router.Route(body.Probs))
}

func (s *Server) certify(w http.ResponseWriter, r *http.Request) {
	if !s.auth(w, r) {
		return
	}
	var body struct {
		Rmin      float64        `json:"rmin"`
		Delta     float64        `json:"delta"`
		Procedure string         `json:"procedure"`
		Policies  []cert.Policy  `json:"policies"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if body.Rmin == 0 {
		body.Rmin = 0.95
	}
	if body.Delta == 0 {
		body.Delta = 0.05
	}
	var passed []cert.Policy
	switch body.Procedure {
	case "bonferroni":
		passed = cert.Bonferroni(body.Policies, body.Rmin, body.Delta)
	case "holm":
		passed = cert.Holm(body.Policies, body.Rmin, body.Delta)
	case "pareto":
		passed = cert.Pareto(cert.OrderPareto(body.Policies), body.Rmin, 40, 3.0, body.Delta)
	default:
		passed = cert.FixedSequence(body.Policies, body.Rmin, body.Delta)
	}
	best, ok := cert.Cheapest(passed)
	writeJSON(w, map[string]any{"passed": passed, "selected": best, "ok": ok})
}

func (s *Server) cr(w http.ResponseWriter, r *http.Request) {
	if !s.auth(w, r) {
		return
	}
	var body struct {
		N1, N2, M, Detected, F1, F2 int
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	nHat := audit.Chapman(body.N1, body.N2, body.M)
	writeJSON(w, map[string]any{
		"chapman": nHat,
		"chao":    audit.Chao(body.Detected, body.F1, body.F2, 4),
		"recall":  audit.Recall(body.Detected, nHat),
	})
}

func (s *Server) auditEvent(w http.ResponseWriter, r *http.Request) {
	if !s.auth(w, r) {
		return
	}
	var body struct {
		Hit bool `json:"hit"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	s.mu.Lock()
	s.audit.Events++
	if body.Hit {
		s.audit.Detected++
	}
	s.mu.Unlock()
	writeJSON(w, s.audit)
}

func (s *Server) mon(w http.ResponseWriter, r *http.Request) {
	if !s.auth(w, r) {
		return
	}
	s.mu.Lock()
	winc := s.audit
	s.mu.Unlock()
	writeJSON(w, map[string]any{"window": winc, "alarm": monitor.Alarm(winc, 0.95, 1.96)})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
