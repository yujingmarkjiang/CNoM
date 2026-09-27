#pragma once
#include "cnom/types.hpp"
#include <deque>
#include <string>

namespace cnom {

class TemporalGate {
 public:
    TemporalGate(std::string camera_id, std::string source_id, TemporalPolicy policy);
    std::vector<CandidateEvent> update(double timestamp_s, float score);
    std::vector<CandidateEvent> flush(double final_timestamp_s = -1);

 private:
    CandidateEvent close(double max_end_s);
    std::string camera_id_;
    std::string source_id_;
    TemporalPolicy policy_;
    std::deque<std::pair<double, float>> window_;
    double active_start_ = -1;
    double active_trigger_ = -1;
    double active_last_ = -1;
    std::vector<float> active_scores_;
    double cooldown_until_ = -1e300;
    int sequence_ = 0;
};

float max_aggregate(const float* scores, int n);

}  // namespace cnom
