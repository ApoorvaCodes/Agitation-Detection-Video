"""Small supervised masked two-centroid classifier, not a probability model."""
from typing import Literal
from pydantic import Field, model_validator
from person2.contracts import Contract, Embedding, Prototype, PrototypeBank
from person2.prototypes import build_prototypes, cosine_similarity
from cmai.taxonomy import VERSION, require_action_items


class ActionCentroids(Contract):
    item_id: str
    positive: Prototype
    negative: Prototype

    @model_validator(mode="after")
    def compatible(self):
        require_action_items([self.item_id])
        a,b = self.positive.embedding, self.negative.embedding
        if a.space != b.space or len(a.values) != len(b.values):
            raise ValueError("positive/negative centroids use incompatible embedding spaces")
        if self.positive.behaviour != self.item_id or self.negative.behaviour != self.item_id:
            raise ValueError("classifier centroids must retain the canonical item ID")
        return self


class ActionModel(Contract):
    schema_version: Literal["cmai-action-model-1.0"] = "cmai-action-model-1.0"
    taxonomy_version: Literal["cmai-long-form-camera-v1"] = VERSION
    model_identity: Literal["masked_two_centroid_classifier_v1"] = "masked_two_centroid_classifier_v1"
    version: str = Field(min_length=1)
    encoder_identity: dict
    classes: list[ActionCentroids] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_classes(self):
        if len({c.item_id for c in self.classes}) != len(self.classes):
            raise ValueError("duplicate action model labels")
        spaces = {(c.positive.embedding.space, len(c.positive.embedding.values)) for c in self.classes}
        if len(spaces) != 1:
            raise ValueError("action classes must share a single versioned embedding space")
        return self

    def positive_bank(self):
        return PrototypeBank(version=self.version, prototypes=[c.positive for c in self.classes])


def train_action_model(examples, labels, version, encoder_identity):
    """examples: (explicit item ID, positive/negative, usable train embedding)."""
    require_action_items(labels)
    examples = list(examples)
    if any(item not in labels or outcome not in {"positive", "negative"} for item,outcome,embedding in examples):
        raise ValueError("only explicitly labelled positive/negative examples for declared actions are allowed")
    classes = []
    for label in labels:
        centroids = []
        for outcome in ("positive", "negative"):
            selected = [(label, embedding) for item, target, embedding in examples if item == label and target == outcome]
            if not selected:
                raise ValueError(f"usable training {outcome} examples required: {label}")
            centroids.append(build_prototypes(selected, version).prototypes[0])
        classes.append(ActionCentroids(item_id=label, positive=centroids[0], negative=centroids[1]))
    return ActionModel(version=version, encoder_identity=encoder_identity, classes=classes)


def score_action(embedding, model_class, min_shared_fraction):
    positive, shared_positive = cosine_similarity(embedding, model_class.positive.embedding, min_shared_fraction)
    negative, shared_negative = cosine_similarity(embedding, model_class.negative.embedding, min_shared_fraction)
    if positive is None or negative is None:
        return None, None, min(shared_positive, shared_negative)
    return positive, positive-negative, min(shared_positive, shared_negative)
