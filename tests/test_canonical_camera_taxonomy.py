import pytest

from cmai.taxonomy import validate_canonical_cmai_behaviour


@pytest.mark.parametrize("item_id,name", [
    ("cmai_07_hitting", "Hitting"),
    ("cmai_08_kicking", "Kicking"),
    ("cmai_01_pacing_aimless_wandering", "Pacing, aimless wandering"),
    ("cmai_26_repetitious_mannerisms", "Repetitious mannerisms"),
    ("cmai_29_general_restlessness", "General restlessness"),
])
def test_canonical_camera_items_resolve_from_taxonomy(item_id, name):
    item = validate_canonical_cmai_behaviour(item_id, name)
    assert item.item_id == item_id and item.display_name == name


@pytest.mark.parametrize("item_id,name", [
    ("gentle restlessness", None),
    ("restlessness", None),
    ("cmai_99_unknown", None),
    (None, None),
    ("cmai_07_hitting", "Kicking"),
])
def test_free_text_unknown_or_mismatched_labels_fail_closed(item_id, name):
    with pytest.raises(ValueError):
        validate_canonical_cmai_behaviour(item_id, name)


def test_audio_only_item_is_not_allowed_for_physical_video():
    with pytest.raises(ValueError):
        validate_canonical_cmai_behaviour("cmai_04_cursing_verbal_aggression")
