package router

import "github.com/cnom/cnom/internal/types"

type Decision string

const (
	Alert    Decision = "alert"
	Suppress Decision = "suppress"
	Review   Decision = "review"
)

type Result struct {
	Decision Decision `json:"decision"`
	Label    int      `json:"label"`
	Conf     float64  `json:"confidence"`
	PSafe    float64  `json:"p_safe"`
	PUnsafe  float64  `json:"p_unsafe"`
}

type Router struct {
	TauS float64
	TauU float64
	TCal float64
}

func New() Router { return Router{TauS: 0.985, TauU: 0.80, TCal: 1.34} }

func (r Router) Route(p []float64) Result {
	sum := 0.0
	for _, v := range p {
		if v < 0 {
			return Result{Decision: Review}
		}
		sum += v
	}
	if sum <= 0 || len(p) < 2 {
		return Result{Decision: Review}
	}
	label := 0
	best := p[0] / sum
	for i := 1; i < len(p); i++ {
		v := p[i] / sum
		if v > best {
			best = v
			label = i
		}
	}
	out := Result{Label: label, Conf: best, PSafe: p[0] / sum, PUnsafe: 1 - p[0]/sum}
	ns := false
	for _, id := range types.NonSuppressible {
		if label == id {
			ns = true
			break
		}
	}
	switch {
	case out.PUnsafe >= r.TauU:
		out.Decision = Alert
	case out.PSafe >= r.TauS && !ns:
		out.Decision = Suppress
	default:
		out.Decision = Review
	}
	return out
}
