from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    model_name: str
    raw_data_dir: Path = Path("data/raw")
    cache_subdir: str = "go_emotions_simplified"
    outputs_dir: Path = Path("outputs")
    eval_outputs_dir: Path = Path("eval_outputs")
    saved_models_dir: Path = Path("saved_models")
    vocab_size: int = 10000
    sequence_length: int = 100
    batch_size: int = 32
    embedding_dim: int = 128
    lstm_units: int = 64
    dropout_rate: float = 0.2
    epochs: int = 20
    early_stopping_patience: int = 3
    threshold: float = 0.5

    @property
    def hf_cache_dir(self) -> Path:
        return self.raw_data_dir / self.cache_subdir

    @property
    def model_output_path(self) -> Path:
        return self.saved_models_dir / f"goemotions_{self.model_name}.keras"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    def ensure_dirs(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
