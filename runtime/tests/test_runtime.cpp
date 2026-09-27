#include "cnom/gate.hpp"
#include "cnom/router.hpp"
#include <cmath>
#include <iostream>

static int fails = 0;

static void expect(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "fail: " << msg << "\n";
        ++fails;
    }
}

int main() {
    float s[8] = {0.1, 0.2, 0.9, 0.3, 0.1, 0.1, 0.1, 0.1};
    expect(std::fabs(cnom::max_aggregate(s, 8) - 0.9f) < 1e-6, "max");

    cnom::TemporalPolicy p;
    cnom::TemporalGate g("cam0", "src0", p);
    std::vector<cnom::CandidateEvent> ev;
    for (int i = 0; i < 80; ++i) {
        const float score = i >= 20 && i < 50 ? 0.8f : 0.05f;
        auto out = g.update(i * 0.1, score);
        ev.insert(ev.end(), out.begin(), out.end());
    }
    auto tail = g.flush(8.0);
    ev.insert(ev.end(), tail.begin(), tail.end());
    expect(!ev.empty(), "gate emitted");
    if (!ev.empty()) {
        expect(ev[0].start_s >= 0, "start");
        expect(ev[0].end_s >= ev[0].trigger_s, "end>=trigger");
        expect(ev[0].max_score > 0.5f, "max score");
    }

    cnom::Router r;
    float safe[9] = {0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003};
    expect(r.route(safe, 9).decision == cnom::Route::Suppress, "suppress");
    float uns[9] = {0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.88};
    expect(r.route(uns, 9).decision == cnom::Route::Alert, "alert");
    float mid[9] = {0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05};
    expect(r.route(mid, 9).decision == cnom::Route::Review, "review");
    float ns_safe[9] = {0.99f, 0.001f, 0.001f, 0.002f, 0.001f, 0.001f, 0.002f, 0.001f, 0.001f};
    expect(r.route(ns_safe, 9).decision == cnom::Route::Suppress, "high-safe suppress");
    float ns_lab[9] = {0.40f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.48f, 0.01f, 0.01f};
    expect(r.route(ns_lab, 9).label == 6, "ns label");
    expect(r.route(ns_lab, 9).decision == cnom::Route::Review, "ns never suppress");

    if (fails) {
        std::cerr << fails << " failures\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
