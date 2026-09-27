package cert

import (
	"math"
	"sort"
)

type Policy struct {
	TauF     float64 `json:"tau_f"`
	W        int     `json:"w"`
	Rho      float64 `json:"rho"`
	Upload   float64 `json:"upload"`
	K        int     `json:"detections"`
	N        int     `json:"events"`
	NEff     float64 `json:"n_eff,omitempty"`
	Burden   float64 `json:"burden,omitempty"`
	Latency  float64 `json:"p95,omitempty"`
	Recall   float64 `json:"recall,omitempty"`
	PR       float64 `json:"p_r,omitempty"`
	PA       float64 `json:"p_a,omitempty"`
	PL       float64 `json:"p_l,omitempty"`
}

func WilsonLower(k, n int, z float64) float64 {
	if n <= 0 || k < 0 || k > n {
		return math.NaN()
	}
	p := float64(k) / float64(n)
	den := 1 + z*z/float64(n)
	center := (p + z*z/(2*float64(n))) / den
	half := z * math.Sqrt((p*(1-p)+z*z/(4*float64(n)))/float64(n)) / den
	lo := center - half
	if lo < 0 {
		return 0
	}
	return lo
}

func PValue(n, k int, rmin float64) float64 {
	return BinomialPGE(n, k, rmin)
}

func PValueEff(n, k int, nEff, rmin float64) float64 {
	if nEff <= 0 || n <= 0 {
		return PValue(n, k, rmin)
	}
	ne := int(math.Round(nEff))
	if ne < 1 {
		ne = 1
	}
	ke := int(math.Round(float64(k) * nEff / float64(n)))
	if ke < 0 {
		ke = 0
	}
	if ke > ne {
		ke = ne
	}
	return BinomialPGE(ne, ke, rmin)
}

func BinomialPGE(n, k int, p0 float64) float64 {
	if n <= 0 || k < 0 || k > n {
		return 1
	}
	if k == 0 {
		return 1
	}
	if p0 <= 0 {
		return 0
	}
	if p0 >= 1 {
		return 1
	}
	logTerm := logChoose(n, k) + float64(k)*math.Log(p0) + float64(n-k)*math.Log(1-p0)
	term := math.Exp(logTerm)
	if term == 0 && k < n {
		return 0
	}
	acc := term
	q := p0 / (1 - p0)
	for i := k; i < n; i++ {
		term *= float64(n-i) / float64(i+1) * q
		acc += term
		if acc > 1 {
			return 1
		}
	}
	return acc
}

func binomPMF(n, k int, p float64) float64 {
	if p <= 0 {
		if k == 0 {
			return 1
		}
		return 0
	}
	if p >= 1 {
		if k == n {
			return 1
		}
		return 0
	}
	logc := logChoose(n, k) + float64(k)*math.Log(p) + float64(n-k)*math.Log(1-p)
	return math.Exp(logc)
}

func logChoose(n, k int) float64 {
	if k < 0 || k > n {
		return math.Inf(-1)
	}
	if k > n-k {
		k = n - k
	}
	s := 0.0
	for i := 1; i <= k; i++ {
		s += math.Log(float64(n-k+i)) - math.Log(float64(i))
	}
	return s
}

func Bonferroni(policies []Policy, rmin, delta float64) []Policy {
	if len(policies) == 0 {
		return nil
	}
	alpha := delta / float64(len(policies))
	var out []Policy
	for _, p := range policies {
		pv := PValueEff(p.N, p.K, p.NEff, rmin)
		if pv <= alpha {
			p.PR = pv
			out = append(out, p)
		}
	}
	return out
}

func Holm(policies []Policy, rmin, delta float64) []Policy {
	type item struct {
		i int
		p float64
	}
	items := make([]item, len(policies))
	for i, pol := range policies {
		items[i] = item{i, PValueEff(pol.N, pol.K, pol.NEff, rmin)}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].p < items[b].p })
	m := len(items)
	keep := make([]bool, m)
	for j, it := range items {
		if it.p > delta/float64(m-j) {
			break
		}
		keep[it.i] = true
	}
	var out []Policy
	for i, pol := range policies {
		if keep[i] {
			pol.PR = PValueEff(pol.N, pol.K, pol.NEff, rmin)
			out = append(out, pol)
		}
	}
	return out
}

func FixedSequence(ordered []Policy, rmin, delta float64) []Policy {
	var out []Policy
	for _, p := range ordered {
		pv := PValueEff(p.N, p.K, p.NEff, rmin)
		p.PR = pv
		if pv > delta {
			break
		}
		out = append(out, p)
	}
	return out
}

func Pareto(ordered []Policy, rmin, amax, lmax, delta float64) []Policy {
	var out []Policy
	for _, p := range ordered {
		p.PR = PValueEff(p.N, p.K, p.NEff, rmin)
		if p.Burden > amax {
			p.PA = 1
		} else {
			p.PA = 0
		}
		if p.Latency > lmax {
			p.PL = 1
		} else {
			p.PL = 0
		}
		if math.Max(p.PR, math.Max(p.PA, p.PL)) > delta {
			break
		}
		out = append(out, p)
	}
	return out
}

func OrderPareto(policies []Policy) []Policy {
	out := append([]Policy(nil), policies...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Recall != out[j].Recall {
			return out[i].Recall > out[j].Recall
		}
		return out[i].Upload > out[j].Upload
	})
	return out
}

func Cheapest(ps []Policy) (Policy, bool) {
	if len(ps) == 0 {
		return Policy{}, false
	}
	best := ps[0]
	for _, p := range ps[1:] {
		if p.Upload < best.Upload {
			best = p
		}
	}
	return best, true
}
