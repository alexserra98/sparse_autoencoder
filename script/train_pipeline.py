import argparse
from src.train import Trainer, Parser, MLPActivation

def main():
    # Parse command line arguments
    config, data_config, trainer_config = Parser.get_config()

    # Initialize the Trainer with the parsed configurations
    trainer = Trainer(config, data_config, trainer_config)

    # Select and initialize an activation
    # For example, using MLPActivation for layer 0
    # You can modify the layer as needed
    mlp_activation = MLPActivation(layer=8)

    # Run the training pipeline
    trainer.run_pipeline(mlp_activation)

if __name__ == "__main__":
    main()