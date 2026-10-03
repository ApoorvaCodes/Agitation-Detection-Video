"""Person 2 temporal embedding and prototype-matching research baseline."""
from person2.config import Person2Config
from person2.pipeline import process_perception
from person2.prototypes import build_prototypes

__all__ = ["Person2Config", "process_perception", "build_prototypes"]
