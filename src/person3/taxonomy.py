"""Centralized CMAI item vocabulary and explicit project-label mappings.

Labels are review categories; they do not assert diagnosis or CMAI frequency.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Behaviour:
    key: str
    label: str
    domain: str
    aliases: tuple[str, ...] = ()


BEHAVIOURS = (
    Behaviour("hitting", "Hitting", "physical"),
    Behaviour("kicking", "Kicking", "physical"),
    Behaviour("pushing", "Pushing", "physical"),
    Behaviour("grabbing", "Grabbing", "physical"),
    Behaviour("scratching", "Scratching", "physical"),
    Behaviour("biting", "Biting", "physical"),
    Behaviour("throwing", "Throwing objects", "physical"),
    Behaviour("tearing_destroying", "Tearing or destroying things", "physical", ("tearing/destroying", "tearing or destroying")),
    Behaviour("spitting", "Spitting", "physical"),
    Behaviour("verbal_aggression_cursing", "Verbal aggression or cursing", "verbal", ("verbal aggression/cursing", "cursing")),
    Behaviour("repetitive_questioning", "Repetitive questioning", "verbal"),
    Behaviour("repetitive_requests", "Repetitive requests", "verbal"),
    Behaviour("complaining", "Complaining", "verbal"),
    Behaviour("negativism_resistance", "Negativism or resistance", "verbal", ("negativism/resistance",)),
    Behaviour("constant_requests_help_attention", "Constant requests for help or attention", "verbal"),
    Behaviour("distressed_urgent_verbalization", "Distressed or urgent verbalization", "verbal"),
    Behaviour("screaming", "Screaming", "verbal"),
    Behaviour("vocal_agitation", "Vocal agitation", "verbal"),
    Behaviour("strange_non_speech_vocal_behaviour", "Strange or non-speech vocal behaviour", "verbal", ("strange/non-speech vocal behaviour",)),
    Behaviour("pacing_aimless_wandering", "Pacing or aimless wandering", "physical", ("pacing_aimless_wandering",)),
    Behaviour("repetitious_mannerisms", "Repetitious mannerisms", "physical"),
    Behaviour("general_restlessness", "General restlessness", "physical"),
)

_LOOKUP = {}
for item in BEHAVIOURS:
    for token in (item.key, item.label, *item.aliases):
        _LOOKUP[token.strip().casefold().replace("-", "_")] = item


def normalize_behaviour(value: str) -> Behaviour | None:
    """Resolve only explicit taxonomy labels/aliases; no fuzzy classification."""
    key = value.strip().casefold().replace("-", "_")
    return _LOOKUP.get(key) or _LOOKUP.get(key.replace("_", " "))


def taxonomy_labels() -> tuple[str, ...]:
    return tuple(item.label for item in BEHAVIOURS)
