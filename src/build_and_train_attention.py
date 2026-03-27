from config import TrainingConfig
from training import run_experiment

if __name__ == "__main__":
    config = TrainingConfig(
        model_name="attention",
        embedding_dim=256,
        lstm_units=64,
        dropout_rate=0.3,
    )
    
    run_experiment(config)
