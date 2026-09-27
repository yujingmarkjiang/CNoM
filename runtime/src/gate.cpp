#include "cnom/gate.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace cnom {

float max_aggregate(const float* scores, int n) {
    if (n <= 0 || !scores) return 0;
    float m = scores[0];
    for (int i = 1; i < n; ++i) m = std::max(m, scores[i]);
    return m;
}

TemporalGate::TemporalGate(std::string camera_id, std::string source_id, TemporalPolicy policy)
    : camera_id_(std::move(camera_id)), source_id_(std::move(source_id)), policy_(policy) {
    if (policy_.window_frames <= 0 || policy_.persistence <= 0 || policy_.persistence > 1)
        throw std::invalid_argument("invalid temporal policy");
}

std::vector<CandidateEvent> TemporalGate::update(double t, float score) {
    if (!std::isfinite(t) || !std::isfinite(score) || score < 0 || score > 1)
        throw std::invalid_argument("invalid observation");
    window_.push_back({t, score});
    while (static_cast<int>(window_.size()) > policy_.window_frames) window_.pop_front();
    std::vector<CandidateEvent> out;
    if (static_cast<int>(window_.size()) < policy_.window_frames) return out;

    int above = 0;
    for (auto& p : window_)
        if (p.second > policy_.frame_threshold) ++above;
    const float a = static_cast<float>(above) / static_cast<float>(window_.size());

    if (a >= policy_.persistence && t >= cooldown_until_) {
        if (active_start_ < 0) {
            active_start_ = window_.front().first;
            active_trigger_ = t;
            active_scores_.clear();
            for (auto& p : window_) active_scores_.push_back(p.second);
        } else {
            active_scores_.push_back(score);
        }
        active_last_ = t;
        return out;
    }
    if (active_start_ >= 0 && a < policy_.persistence) {
        if (t - active_last_ <= policy_.merge_seconds) return out;
        out.push_back(close(active_last_));
    }
    return out;
}

std::vector<CandidateEvent> TemporalGate::flush(double final_t) {
    if (active_start_ < 0) return {};
    return {close(final_t >= 0 ? final_t : active_last_)};
}

CandidateEvent TemporalGate::close(double close_t) {
    if (active_start_ < 0 || active_last_ < 0 || active_trigger_ < 0)
        throw std::runtime_error("no active event");
    const double end = std::min(active_start_ + policy_.max_clip_s,
                                std::max(active_trigger_, close_t + policy_.post_roll_s));
    ++sequence_;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d", sequence_);
    float mx = 0, sum = 0;
    int above = 0;
    for (float s : active_scores_) {
        mx = std::max(mx, s);
        sum += s;
        if (s > policy_.frame_threshold) ++above;
    }
    CandidateEvent e;
    e.id = source_id_ + ":" + buf;
    e.camera_id = camera_id_;
    e.source_id = source_id_;
    e.start_s = std::max(0.0, active_start_ - policy_.pre_roll_s);
    e.end_s = end;
    e.trigger_s = active_trigger_;
    e.max_score = mx;
    e.mean_score = active_scores_.empty() ? 0 : sum / static_cast<float>(active_scores_.size());
    e.frames_above = above;
    e.frames_total = static_cast<int>(active_scores_.size());
    cooldown_until_ = close_t + policy_.cooldown_seconds;
    active_start_ = active_trigger_ = active_last_ = -1;
    active_scores_.clear();
    return e;
}

}  // namespace cnom
