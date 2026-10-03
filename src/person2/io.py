import json
from pathlib import Path

from person2.contracts import Person2VideoResult, PrototypeBank


def save_result(result: Person2VideoResult, path):
    Path(path).write_text(json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=False) + "\n")


def load_result(path):
    return Person2VideoResult.model_validate_json(Path(path).read_text())


def save_prototypes(bank: PrototypeBank, path):
    Path(path).write_text(json.dumps(bank.model_dump(mode="json"), indent=2, allow_nan=False) + "\n")


def load_prototypes(path):
    return PrototypeBank.model_validate_json(Path(path).read_text())
