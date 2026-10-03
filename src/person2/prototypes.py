from collections import defaultdict
import numpy as np

from person2.contracts import Embedding, Prototype, PrototypeBank


def cosine_similarity(a: Embedding, b: Embedding, min_shared_fraction=0.5):
    if a.space != b.space or len(a.values) != len(b.values):
        raise ValueError("incompatible embedding spaces; rebuild prototypes with the same encoder/configuration")
    mask = np.asarray(a.valid) & np.asarray(b.valid)
    union = np.asarray(a.valid) | np.asarray(b.valid)
    fraction = float(mask.sum() / union.sum()) if union.any() else 0.0
    if not mask.any() or fraction < min_shared_fraction:
        return None, fraction
    av, bv = np.asarray(a.values)[mask], np.asarray(b.values)[mask]
    norm = np.linalg.norm(av) * np.linalg.norm(bv)
    if norm <= 1e-12:
        return None, fraction
    return float(np.clip(np.dot(av, bv) / norm, -1, 1)), fraction


def build_prototypes(examples: list[tuple[str, Embedding]], version: str):
    """Masked centroid of labelled exemplars; no text/pose space mixing."""
    grouped = defaultdict(list)
    for behaviour, embedding in examples:
        if not isinstance(behaviour, str) or not behaviour.strip():
            raise ValueError("prototype examples require an explicit nonempty behaviour label")
        grouped[behaviour].append(embedding)
    prototypes = []
    for behaviour, embeddings in sorted(grouped.items()):
        reference = embeddings[0]
        for embedding in embeddings:
            if embedding.space != reference.space or len(embedding.values) != len(reference.values):
                raise ValueError("all exemplars for a behaviour must share one embedding space")
            if not any(embedding.valid):
                raise ValueError("prototype exemplars must contain valid evidence")
        vectors = np.asarray([e.values for e in embeddings])
        masks = np.asarray([e.valid for e in embeddings])
        # Normalize before averaging so exemplar magnitude cannot dominate.
        norms = np.linalg.norm(np.where(masks, vectors, 0), axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("prototype exemplar has zero norm")
        vectors = vectors / norms[:, None]
        counts = masks.sum(axis=0)
        center = np.divide(np.where(masks, vectors, 0).sum(axis=0), counts,
                           out=np.zeros(vectors.shape[1]), where=counts > 0)
        if np.linalg.norm(center) <= 1e-12:
            raise ValueError("prototype centroid cancels to zero")
        prototypes.append(Prototype(behaviour=behaviour, example_count=len(embeddings),
                                   embedding=Embedding(space=reference.space, values=center.tolist(),
                                                       valid=(counts > 0).tolist())))
    return PrototypeBank(version=version, prototypes=prototypes)
