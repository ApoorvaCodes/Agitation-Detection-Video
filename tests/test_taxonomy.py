from person3.taxonomy import BEHAVIOURS, normalize_behaviour, taxonomy_labels
from cmai.taxonomy import camera_observable_ids


def test_taxonomy_is_central_and_aliases_resolve():
    assert len(taxonomy_labels()) == len(BEHAVIOURS)
    assert normalize_behaviour("verbal aggression/cursing").key == "verbal_aggression_cursing"
    assert normalize_behaviour("unknown clinical label") is None
    assert {x.domain for x in BEHAVIOURS} == {"physical", "verbal"}


def test_camera_observable_ids_exclude_audio_and_context_items():
    ids = camera_observable_ids()
    assert "cmai_01_pacing_aimless_wandering" in ids
    assert "cmai_26_repetitious_mannerisms" in ids
    assert "cmai_04_cursing_verbal_aggression" not in ids
    assert "cmai_06_repetitive_sentences_questions" not in ids
    assert "cmai_20_inappropriate_substance_ingestion" not in ids
