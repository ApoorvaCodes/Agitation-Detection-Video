"""Small orchestration boundary for the video-first dashboard."""
import json
import os

from person1.config import Person1Config
from person1.contracts import Person1VideoResult
from person2.config import Person2Config
from person3.p2_adapter import read_p2_handoff
from person2.contracts import Person2VideoResult


def resolve_groq_key(entered="", configured=""):
    """User entry wins; configured secrets and environment remain supported."""
    return entered.strip() or str(configured or "").strip() or os.getenv("GROQ_API_KEY", "").strip()


def analyze_video(path, prototypes=None, p1_config=None, p2_config=None,
                  perception_runner=None, candidate_runner=None):
    if perception_runner is None:
        from person1.pipeline import process_video
        perception_runner = process_video
    if candidate_runner is None:
        from person2.pipeline import process_perception
        candidate_runner = process_perception
    p1 = perception_runner(path, config=p1_config or Person1Config())
    p2 = candidate_runner(p1, config=p2_config or Person2Config(), prototypes=prototypes)
    return p1, p2


def read_dashboard_results(p1_data, p2_data):
    p1 = Person1VideoResult.model_validate_json(p1_data)
    raw = json.loads(p2_data)
    if raw.get("video_id") != p1.video.video_id:
        raise ValueError("Person 1 and Person 2 results must be from the same video")
    return p1, Person2VideoResult.model_validate(raw)
