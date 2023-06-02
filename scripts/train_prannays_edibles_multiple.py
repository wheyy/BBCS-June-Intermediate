from train_prannays_edibles import train_prannays_edibles
import logging

logging.basicConfig(level=logging.INFO)

def main():
    wandb_api_key = input("Enter wandb API key: ")

    epochs = 20
    params_to_try = [
        {
            'train_split_percentage': 0.6,
        },
        {
            'train_split_percentage': 0.8,
        },
        {
            'batch': 1,
        },
    ]
    for params_dict in params_to_try:
        logging.info("Starting try for params: %s", params_dict)
        train_prannays_edibles(wandb_api_key, epochs=epochs, recreate_split_dataset=True, **params_dict)


if __name__ == '__main__':
    main()
