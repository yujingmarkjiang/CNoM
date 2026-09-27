#pragma once
#include "cnom/types.hpp"
#include <span>

namespace cnom {

class Router {
 public:
    Router(float tau_s = 0.985f, float tau_u = 0.80f);
    RouteResult route(const float* probs, int n) const;

 private:
    float tau_s_;
    float tau_u_;
};

}  // namespace cnom
