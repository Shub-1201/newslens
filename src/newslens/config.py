"""Every setting that shapes the model, in one place."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "model"
RESULTS_DIR = ROOT / "artifacts" / "results"
CACHE_DIR = ROOT / ".cache"
FEEDBACK_FILE = ROOT / "data" / "feedback" / "feedback.csv"

BASE_MODEL = "google-bert/bert-base-uncased"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    max_length: int = 128
    trainable_layers: int = 2
    batch_size: int = 16
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    huffpost_per_topic: int = 600
    min_words: int = 5
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = TrainConfig()
