package match

func Overlap(a0, a1, b0, b1 float64) float64 {
	lo, hi := a0, a1
	if b0 > lo {
		lo = b0
	}
	if b1 < hi {
		hi = b1
	}
	inter := hi - lo
	if inter <= 0 {
		return 0
	}
	union := (a1 - a0) + (b1 - b0) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func OnsetOK(pred, ref, tol float64) bool {
	d := pred - ref
	if d < 0 {
		d = -d
	}
	return d <= tol
}
