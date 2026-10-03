from person3.taxonomy import BEHAVIOURS, normalize_behaviour, taxonomy_labels


def test_taxonomy_is_central_and_aliases_resolve():
    assert len(taxonomy_labels()) == len(BEHAVIOURS)
    assert normalize_behaviour("verbal aggression/cursing").key == "verbal_aggression_cursing"
    assert normalize_behaviour("unknown clinical label") is None
    assert {x.domain for x in BEHAVIOURS} == {"physical", "verbal"}
