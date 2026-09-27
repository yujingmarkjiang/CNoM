package cascade

import "math"

type Scores struct {
	PUnsafe float64
	Entropy float64
	Margin  float64
	Conf    float64
	Diff    float64
}

func Entropy(p []float64) float64 {
	s := 0.0
	for _, v := range p {
		if v > 0 {
			s -= v * math.Log(v)
		}
	}
	return s
}

func Margin(p []float64) float64 {
	if len(p) < 2 {
		return 0
	}
	a, b := p[0], math.Inf(-1)
	if p[1] > a {
		a, b = p[1], a
	} else {
		b = p[1]
	}
	for i := 2; i < len(p); i++ {
		if p[i] > a {
			b = a
			a = p[i]
		} else if p[i] > b {
			b = p[i]
		}
	}
	return a - b
}

func Rank(kind string, s Scores) float64 {
	switch kind {
	case "random":
		return s.Conf
	case "confidence":
		return 1 - s.Conf
	case "entropy":
		return s.Entropy
	case "margin":
		return -s.Margin
	case "l2d":
		return s.PUnsafe
	case "topk":
		return s.PUnsafe
	case "noscope":
		return s.Diff
	case "reducto":
		return s.Diff
	default:
		return s.PUnsafe
	}
}

func SelectBudget(ranks []float64, frac float64) []bool {
	n := len(ranks)
	k := int(math.Round(frac * float64(n)))
	if k < 0 {
		k = 0
	}
	if k > n {
		k = n
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if ranks[idx[j]] > ranks[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	out := make([]bool, n)
	for i := 0; i < k; i++ {
		out[idx[i]] = true
	}
	return out
}
