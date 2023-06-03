import logging

import wandb

from prepare_data import prepare_data
from common import prannays_edibles_class_name_map, food11_class_name_map, prannays_edibles_dataset_path, food11_dataset_path, classifier_project_name, init_wandb, init_model, train_model

logging.basicConfig(level=logging.INFO)

def train_prannays_edibles(wandb_api_key: str, epochs=15, batch=4, wandb_enabled=True, model_size='n', augmentation_params={}):
    dataset_path = prepare_data(
        prannays_edibles_dataset_path,
        prannays_edibles_class_name_map,
        food11_dataset_path,
        food11_class_name_map
    )

    model = init_model(model_size, model_type='-cls', reset=True)

    init_wandb(wandb_api_key, classifier_project_name, wandb_enabled)

    train_model(
        model=model,
        dataset_path=dataset_path,
        epochs=epochs,
        batch=batch,
        save_period=5,
        augmentation_params=augmentation_params,
    )

    wandb.finish()


if __name__ == '__main__':
    wandb_api_key = input("Enter wandb API key: ")
    wandb_enabled = wandb_api_key != 'no'
    train_prannays_edibles(wandb_api_key, wandb_enabled=wandb_enabled, augmentation_params={'augment': True})
