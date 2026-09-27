#include "cnom/router.hpp"
#include <stdexcept>

namespace cnom {

Router::Router(float tau_s, float tau_u) : tau_s_(tau_s), tau_u_(tau_u) {
    if (tau_s_ < 0 || tau_s_ > 1 || tau_u_ < 0 || tau_u_ > 1)
        throw std::invalid_argument("router thresholds");
}

RouteResult Router::route(const float* p, int n) const {
    if (!p || n < 2) throw std::invalid_argument("probability vector");
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        if (p[i] < 0) throw std::invalid_argument("negative probability");
        sum += p[i];
    }
    if (sum <= 0) throw std::invalid_argument("zero mass");
    int label = 0;
    float best = p[0] / sum;
    for (int i = 1; i < n; ++i) {
        const float v = p[i] / sum;
        if (v > best) {
            best = v;
            label = i;
        }
    }
    RouteResult r;
    r.label = label;
    r.confidence = best;
    r.p_safe = p[0] / sum;
    r.p_unsafe = 1.0f - r.p_safe;
    bool ns = false;
    for (int i = 0; i < kNumNonSuppressible; ++i)
        if (label == kNonSuppressible[i]) ns = true;
    if (r.p_unsafe >= tau_u_)
        r.decision = Route::Alert;
    else if (r.p_safe >= tau_s_ && !ns)
        r.decision = Route::Suppress;
    else
        r.decision = Route::Review;
    return r;
}

}  // namespace cnom
