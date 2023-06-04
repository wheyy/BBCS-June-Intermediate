from train_prannays_edibles import train_prannays_edibles
import logging

logging.basicConfig(level=logging.INFO)

def main():
    wandb_api_key = input("Enter wandb API key: ")
    wandb_enabled = wandb_api_key != 'no'

    epochs = 20
    params_to_try = [
        {
            'model_size': 'n',
            'batch': 2,
        },
    ]
    for params_dict in params_to_try:
        logging.info("Starting try for params: %s", params_dict)
        train_prannays_edibles(wandb_api_key, epochs=epochs, wandb_enabled=wandb_enabled, **params_dict)


if __name__ == '__main__':
    main()
