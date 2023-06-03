
from train_object_detection import train_object_detection
import logging

logging.basicConfig(level=logging.INFO)

def main():
    wandb_api_key = input("Enter wandb API key: ")
    wandb_enabled = wandb_api_key != 'no'

    epochs = 15
    params_to_try = [
        {
            'batch': 128,
        },
        {
            'batch': 64,
        },
        {
            'batch': 32,
        },
        {
            'batch': 16,
        },
    ]
    for params_dict in params_to_try:
        logging.info("Starting try for params: %s", params_dict)
        train_object_detection(wandb_api_key, epochs=epochs, wandb_enabled=wandb_enabled, **params_dict)


if __name__ == '__main__':
    main()
