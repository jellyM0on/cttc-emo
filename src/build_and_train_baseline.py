from config import TrainingConfig
from training import run_experiment

if __name__ == "__main__":
    config = TrainingConfig(
        model_name="baseline",
        embedding_dim=128,
        lstm_units=64,
        dropout_rate=0.2,
    )
    
    run_experiment(config)
