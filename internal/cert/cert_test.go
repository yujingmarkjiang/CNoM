package cert_test

import (
	"math"
	"testing"

	"github.com/cnom/cnom/internal/audit"
	"github.com/cnom/cnom/internal/cert"
	"github.com/cnom/cnom/internal/decomp"
	"github.com/cnom/cnom/internal/router"
)

func TestWilson(t *testing.T) {
	lo := cert.WilsonLower(1224, 1264, 1.96)
	if lo < 0.956 || lo > 0.978 {
		t.Fatalf("wilson lower=%v", lo)
	}
}

func TestPValueExact(t *testing.T) {
	// P(Bin(5,0.5) >= 5) = 1/32
	p := cert.PValue(5, 5, 0.5)
	if math.Abs(p-1.0/32.0) > 1e-12 {
		t.Fatalf("p=%v", p)
	}
	if cert.PValue(10, 0, 0.95) != 1 {
		t.Fatalf("k=0 should be 1")
	}
}

func TestBinomialRejectsDeployedRecall(t *testing.T) {
	p := cert.PValue(720, int(math.Round(0.968*720)), 0.95)
	if p > 0.05 {
		t.Fatalf("expected reject at 0.968 n=720, p=%v", p)
	}
}

func TestRouterPaperOrder(t *testing.T) {
	r := router.New()
	if r.Route([]float64{0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003}).Decision != router.Suppress {
		t.Fatal("high safe")
	}
	if r.Route([]float64{0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.88}).Decision != router.Alert {
		t.Fatal("high unsafe")
	}
	got := r.Route([]float64{0.40, 0.02, 0.02, 0.02, 0.02, 0.02, 0.48, 0.01, 0.01})
	if got.Label != 6 || got.Decision != router.Review {
		t.Fatalf("ns must review got=%v", got)
	}
}

func TestChapmanChao(t *testing.T) {
	n := audit.Chapman(10, 10, 9)
	if math.Abs(n-11.1) > 1e-9 {
		t.Fatalf("chapman %v want 11.1", n)
	}
	chao := audit.Chao(100, 10, 5, 4)
	want := 100 + 0.75*(100.0/10.0)
	if math.Abs(chao-want) > 1e-9 {
		t.Fatalf("chao %v want %v", chao, want)
	}
}

func TestDecomp(t *testing.T) {
	r := decomp.Recall(0.978, 0.80, 0.15, 0.90)
	if math.Abs(r-0.978*(0.80+0.15*0.90)) > 1e-12 {
		t.Fatalf("%v", r)
	}
}

func TestFixedSequenceStops(t *testing.T) {
	ps := []cert.Policy{
		{K: 700, N: 720, Upload: 0.12},
		{K: 680, N: 720, Upload: 0.07},
		{K: 650, N: 720, Upload: 0.04},
	}
	got := cert.FixedSequence(ps, 0.95, 0.05)
	if len(got) == 0 {
		t.Fatal("expected at least first policy")
	}
}
