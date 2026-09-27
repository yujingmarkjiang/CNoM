#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cnom {

inline constexpr const char* kLabels[] = {
    "safe_background",
    "no_fall_protection",
    "leaning_on_machine",
    "near_moving_parts",
    "missing_safety_guard",
    "loose_clothing",
    "under_suspended_load",
    "machine_vibration",
    "unsafe_tool_posture",
};
inline constexpr int kNumLabels = 9;
inline constexpr int kNonSuppressible[] = {3, 6};
inline constexpr int kNumNonSuppressible = 2;

struct TemporalPolicy {
    float frame_threshold = 0.30f;
    int window_frames = 20;
    float persistence = 0.55f;
    float merge_seconds = 3.0f;
    float cooldown_seconds = 8.0f;
    float pre_roll_s = 1.6f;
    float post_roll_s = 1.6f;
    float max_clip_s = 12.0f;
};

struct CandidateEvent {
    std::string id;
    std::string camera_id;
    std::string source_id;
    double start_s = 0;
    double end_s = 0;
    double trigger_s = 0;
    float max_score = 0;
    float mean_score = 0;
    int frames_above = 0;
    int frames_total = 0;
};

enum class Route : uint8_t { Alert, Suppress, Review };

struct RouteResult {
    Route decision = Route::Review;
    int label = 0;
    float confidence = 0;
    float p_safe = 0;
    float p_unsafe = 0;
};

}  // namespace cnom
