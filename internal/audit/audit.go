package audit

import "math"

func Chapman(n1, n2, m int) float64 {
	if m < 0 || n1 < 0 || n2 < 0 {
		return math.NaN()
	}
	return float64(n1+1)*float64(n2+1)/float64(m+1) - 1
}

func Chao(sObs, f1, f2, k int) float64 {
	if f2 <= 0 || k <= 1 {
		return math.NaN()
	}
	return float64(sObs) + (float64(k-1)/float64(k))*(float64(f1*f1)/(2*float64(f2)))
}

func Recall(detected int, nHat float64) float64 {
	if nHat <= 0 {
		return math.NaN()
	}
	return float64(detected) / nHat
}

func LogLinearN(u0 float64, nObs int) float64 {
	return float64(nObs) + math.Exp(u0)
}
