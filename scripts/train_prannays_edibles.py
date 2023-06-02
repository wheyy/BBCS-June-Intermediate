import os
from pathlib import Path
import shutil
import logging

from torch.utils.data import random_split
from torchvision import datasets

from ultralytics import YOLO

import wandb

cwd = Path.cwd()

def train_prannays_edibles(wandb_api_key: str, epochs=15, batch=4, train_split_percentage=0.7, recreate_split_dataset=False):
    class_name_map = {
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

    datasets_path = cwd / "datasets"
    prannays_edibles_path = datasets_path / "prannays_edibles"

    split_dataset_path = create_image_classify_data_split_folder(prannays_edibles_path, class_name_map, train_split_percentage=train_split_percentage, recreate=recreate_split_dataset)

    model = init_model(reset=True)

    project_name = 'BuildingBloCS Prannays Edibles Classifier'
    init_wandb(wandb_api_key, project_name)

    train_model(
        model=model,
        dataset_path=split_dataset_path,
        epochs=epochs,
        batch=batch,
        save_period=5,
    )

    wandb.finish()


def train_model(model: YOLO, dataset_path: str, epochs: int, batch: int, save_period: int):
    logging.info("train_model running")
    model.train(data=dataset_path, batch=batch, epochs=epochs, save_period=save_period)


def init_wandb(api_key: str, project_name: str):
    logging.info("init_wandb running")
    os.environ["WANDB_API_KEY"] = api_key
    wandb.init(project=project_name, settings=wandb.Settings(start_method="spawn"), mode='disabled')


def init_model(reset=True):
    logging.info("init_model running")
    model_path = Path('yolov8n-cls.pt')
    if reset and model_path.exists():
        model_path.unlink()
    model = YOLO('yolov8n-cls.pt') # load pretrained model
    return model


def create_image_classify_data_split_folder(not_yet_split_dataset_path: Path, class_name_map: dict, train_split_percentage: float = 0.7, recreate=True):
    logging.info("create_image_classify_data_split_folder running")
    split_dataset_path = not_yet_split_dataset_path.parent / f"{not_yet_split_dataset_path.name}_split"

    if split_dataset_path.exists() and split_dataset_path.is_dir():
        if not recreate:
            return split_dataset_path
    
        shutil.rmtree(split_dataset_path) # reset split

    split_dataset_path.mkdir(exist_ok=True)

    train_dataset_path = split_dataset_path / "train"
    train_dataset_path.mkdir(exist_ok=True)

    test_dataset_path = split_dataset_path / "test"
    test_dataset_path.mkdir(exist_ok=True)

    # generate new split
    dataset = datasets.ImageFolder(root=not_yet_split_dataset_path)

    test_split_percentage = 1 - train_split_percentage

    train_dataset, test_dataset = random_split(dataset, [train_split_percentage, test_split_percentage])

    def get_class_name(class_index):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        return class_name_map[idx_to_class[class_index]]


    def create_dataset_folder(dataset, dataset_path):
        for i, (image, image_class_idx) in enumerate(dataset):
            image_class_name = get_class_name(image_class_idx)
            class_path = dataset_path / str(image_class_name)
            if not class_path.exists():
                class_path.mkdir(exist_ok=True)
            image.save(class_path / f"{image_class_name}_{i}.jpg")


    create_dataset_folder(train_dataset, train_dataset_path)
    create_dataset_folder(test_dataset, test_dataset_path)

    return split_dataset_path


if __name__ == '__main__':
    train_prannays_edibles()
