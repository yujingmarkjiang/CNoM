package monitor

import "github.com/cnom/cnom/internal/cert"

type Window struct {
	Detected int
	Events   int
}

func Alarm(w Window, rmin, z float64) bool {
	if w.Events <= 0 {
		return false
	}
	return cert.WilsonLower(w.Detected, w.Events, z) < rmin
}
