import os
from pathlib import Path
import logging

import torch
from ultralytics import YOLO
import wandb

from prepare_data import prepare_data
from common import prannays_edibles_class_name_map, food11_class_name_map, prannays_edibles_path, food11_path, project_name

logging.basicConfig(level=logging.INFO)

def train_prannays_edibles(wandb_api_key: str, epochs=15, batch=4, wandb_enabled=True, model_size='n', augmentation_params={}):
    dataset_path = prepare_data(
        prannays_edibles_path,
        prannays_edibles_class_name_map,
        food11_path,
        food11_class_name_map
    )

    model = init_model(model_size, reset=True)

    init_wandb(wandb_api_key, project_name, wandb_enabled)

    train_model(
        model=model,
        dataset_path=dataset_path,
        epochs=epochs,
        batch=batch,
        save_period=5,
        augmentation_params=augmentation_params,
    )

    wandb.finish()


def train_model(model: YOLO, dataset_path: str, epochs: int, batch: int, save_period: int, augmentation_params: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("train_model running with device %s", device)
    model.train(data=dataset_path, batch=batch, epochs=epochs, save_period=save_period, device=device.index, workers=16, **augmentation_params)


def init_wandb(api_key: str, project_name: str, enabled=True):
    logging.info("init_wandb running")
    os.environ["WANDB_API_KEY"] = api_key
    wandb.init(project=project_name, settings=wandb.Settings(start_method="spawn"), mode='online' if enabled else 'disabled')


def init_model(model_size: str, reset=True):
    logging.info("init_model running")
    model_name = f'yolov8{model_size}-cls.pt'
    model_path = Path(model_name)
    if reset and model_path.exists():
        model_path.unlink()
    model = YOLO(model_name) # load pretrained model
    return model


if __name__ == '__main__':
    wandb_api_key = input("Enter wandb API key: ")
    wandb_enabled = wandb_api_key != 'no'
    train_prannays_edibles(wandb_api_key, wandb_enabled=wandb_enabled, augmentation_params={'augment': True})
