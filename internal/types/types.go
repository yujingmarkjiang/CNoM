package types

const (
	NumLabels = 9
)

var Labels = []string{
	"safe_background",
	"no_fall_protection",
	"leaning_on_machine",
	"near_moving_parts",
	"missing_safety_guard",
	"loose_clothing",
	"under_suspended_load",
	"machine_vibration",
	"unsafe_tool_posture",
}

var NonSuppressible = []int{3, 6}

type Candidate struct {
	ID        string  `json:"id"`
	CameraID  string  `json:"camera_id"`
	SourceID  string  `json:"source_id"`
	StartS    float64 `json:"start_s"`
	EndS      float64 `json:"end_s"`
	TriggerS  float64 `json:"trigger_s"`
	MaxScore  float64 `json:"max_score"`
	MeanScore float64 `json:"mean_score"`
}

type ClipJob struct {
	Candidate
	ObjectKey string    `json:"object_key"`
	Probs     []float64 `json:"probs,omitempty"`
}
