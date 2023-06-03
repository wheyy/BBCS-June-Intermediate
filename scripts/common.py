from pathlib import Path
import os
import logging

import wandb
import torch
from ultralytics import YOLO

classifier_project_name = 'BuildingBloCS Prannays Edibles Classifier'
detection_project_name = 'BuildingBloCS UECFOOD100 Object Detection'

prannays_edibles_class_name_map = {
    '0': 'bread',
    '1': 'dairy',
    '2': 'dessert',
    '3': 'egg',
    '4': 'fried',
    '5': 'meat',
    '6': 'pasta',
    '7': 'rice',
    '8': 'seafood',
    '9': 'soup',
    '10': 'vegetables',
}

food11_class_name_map = {
    'Bread': 'bread',
    'Dairy product': 'dairy',
    'Dessert': 'dessert',
    'Egg': 'egg',
    'Fried food': 'fried',
    'Meat': 'meat',
    'Noodles-Pasta': 'pasta',
    'Rice': 'rice',
    'Seafood': 'seafood',
    'Soup': 'soup',
    'Vegetable-Fruit': 'vegetables'
}

cwd = Path.cwd()

datasets_path = cwd / "datasets"
prannays_edibles_dataset_path = datasets_path / "prannays_edibles"
food11_dataset_path = datasets_path / "food11"
object_detection_dataset_path = datasets_path / "UECFOOD100"

def init_wandb(api_key: str, project_name: str, enabled=True):
    logging.info("init_wandb running")
    os.environ["WANDB_API_KEY"] = api_key
    wandb.init(project=project_name, settings=wandb.Settings(start_method="spawn"), mode='online' if enabled else 'disabled')


def train_model(model: YOLO, dataset_path: str, epochs: int, batch: int, save_period: int, augmentation_params: dict, single_cls=None, imgsz=None, val=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("train_model running with device %s", device)
    model.train(data=dataset_path, batch=batch, epochs=epochs, save_period=save_period, device=device.index, workers=16, single_cls=single_cls, imgsz=imgsz, val=val, **augmentation_params)


def init_model(model_size: str, model_type='', reset=True):
    logging.info("init_model running")
    model_name = f'yolov8{model_size}{model_type}.pt'
    model_path = Path(model_name)
    if reset and model_path.exists():
        model_path.unlink()
    model = YOLO(model_name) # load pretrained model
    return model
