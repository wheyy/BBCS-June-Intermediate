
from train_object_detection import train_object_detection
from common import finetuning_hyp
import logging

logging.basicConfig(level=logging.INFO)

def main():
    wandb_api_key = input("Enter wandb API key: ")
    wandb_enabled = wandb_api_key != 'no'

    epochs = 20
    params_to_try = [
        {
            'batch': 12,
            'augmentation_params': finetuning_hyp
        },
    ]
    for params_dict in params_to_try:
        logging.info("Starting try for params: %s", params_dict)
        train_object_detection(wandb_api_key, epochs=epochs, wandb_enabled=wandb_enabled, **params_dict)


if __name__ == '__main__':
    main()
