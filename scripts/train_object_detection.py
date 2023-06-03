import shutil
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)

import pandas as pd
from torch.utils.data import random_split
from torchvision import datasets
import yaml
from PIL import Image
import wandb

from common import object_detection_dataset_path, datasets_path, detection_project_name, init_model, init_wandb, train_model

def train_object_detection(wandb_api_key: str, epochs=15, batch=4, single_cls=True, wandb_enabled=True, model_size='n', augmentation_params={}, recreate_dataset=False):
    dataset_yaml_path, prepared_dataset_path = prepare_data(recreate=recreate_dataset)
    model = init_model(model_size, reset=True)
    init_wandb(wandb_api_key, detection_project_name, wandb_enabled)
    train_model(
        model=model,
        dataset_path=dataset_yaml_path,
        epochs=epochs,
        batch=batch,
        save_period=5,
        augmentation_params=augmentation_params,
        single_cls=single_cls,
    )


def prepare_data(train_split_percentage=0.7, recreate=False):
    logging.info("running prepare_data")

    category_dict = get_category_dict()
    prepared_dataset_path = datasets_path / f'{object_detection_dataset_path.name}_prepared'
    dataset_yaml_path = prepared_dataset_path / f'{object_detection_dataset_path.name}.yaml'

    if prepared_dataset_path.exists():
        if recreate:
            shutil.rmtree(prepared_dataset_path)
        else:
            return dataset_yaml_path, prepared_dataset_path
    
    prepared_dataset_path.mkdir(exist_ok=True)

    train_dataset_path = prepared_dataset_path / 'train'
    val_dataset_path = prepared_dataset_path / 'val'

    dataset_yaml_dict = {
        'train': str(train_dataset_path.absolute()),
        'val': str(val_dataset_path.absolute()),
        'names': category_dict,
    }

    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml_dict, f)

    logging.info("loading dataset information into memory")
    dataset = datasets.DatasetFolder(root=object_detection_dataset_path, loader=lambda path: path, extensions=('jpg',))

    class_bounding_box_info = {}
    image_classes_map = {}
    image_sizes_map = {}

    for i, (image_path_str, _) in enumerate(dataset):
        image_path = Path(image_path_str)
        image_name = image_path.stem
        class_path = image_path.parent
        class_index = int(class_path.name) - 1

        if image_name not in image_classes_map:
            image_classes_map[image_name] = []
        
        image_classes_map[image_name].append(class_index)

        if class_index not in class_bounding_box_info:
            bb_info_df = pd.read_csv(class_path / 'bb_info.txt', delim_whitespace=True)
            class_bounding_box_info[class_index] = bb_info_df
        
        if image_name not in image_sizes_map:
            im = Image.open(image_path)
            image_sizes_map[image_name] = im.size

    logging.info("writing data into prepared dataset")

    tmp_path = prepared_dataset_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)

    already_added_image_names = set()

    for i, (image_path_str, _) in enumerate(dataset):
        image_path = Path(image_path_str)
        image_name = image_path.stem

        if image_name in already_added_image_names:
            continue
        else:
            already_added_image_names.add(image_name)
        
        class_path = image_path.parent
        class_index = int(class_path.name) - 1

        new_image_name = f'{i}_{image_name}'
        new_image_path = tmp_path / f'{new_image_name}.jpg'
        shutil.copyfile(image_path, new_image_path)

        image_txt_file_path = new_image_path.parent / f'{new_image_name}.txt'
        image_txt_lines = []

        for other_class_index in image_classes_map[image_name]:
            bb_info_df = class_bounding_box_info[other_class_index]
            image_bb_row = bb_info_df.loc[bb_info_df['img'] == int(image_name)]
            
            x1, y1, x2, y2 = image_bb_row.iloc[0, 1:].to_numpy()
            box_width = x2 - x1
            box_height = y2 - y1

            center_x = x2 - box_width / 2
            center_y = y2 - box_height / 2

            image_width, image_height = image_sizes_map[image_name]
            
            x_normalised = center_x / image_width
            y_normalised = center_y / image_height
            width_normalised = box_width / image_width
            height_normalised = box_height / image_height

            image_txt_lines.append(f'{other_class_index} {x_normalised:.6f} {y_normalised:.6f} {width_normalised:.6f} {height_normalised:.6f}\n')

        with open(image_txt_file_path, 'w') as txt_file:
            txt_file.writelines(image_txt_lines)

    logging.info("splitting prepared dataset into train/val")

    prepared_dataset = datasets.DatasetFolder(root=prepared_dataset_path, loader=lambda path: path, extensions=('jpg'))
    test_split_percentage = 1 - train_split_percentage
    train_dataset, val_dataset = random_split(prepared_dataset, [train_split_percentage, test_split_percentage])

    def move_dataset(specific_dataset, specific_dataset_path):
        for (image_path_str, _) in specific_dataset:
            image_path = Path(image_path_str)
            txt_file_path = image_path.parent / f'{image_path.stem}.txt'

            shutil.move(str(image_path), specific_dataset_path)
            shutil.move(str(txt_file_path), specific_dataset_path)
    
    train_dataset_path.mkdir(exist_ok=True)
    val_dataset_path.mkdir(exist_ok=True)

    move_dataset(train_dataset, train_dataset_path)
    move_dataset(val_dataset, val_dataset_path)

    return dataset_yaml_path, prepared_dataset_path


def get_category_dict():
    category_df = pd.read_csv(object_detection_dataset_path / 'category.txt', sep='\t')
    category_names = category_df['name'].to_list()
    category_index_dict = {i: category_name for i, category_name in enumerate(category_names)}
    return category_index_dict


if __name__ == '__main__':
    wandb_api_key = input("Enter wandb API key: ")
    wandb_enabled = wandb_api_key != 'no'
    train_object_detection(wandb_api_key, wandb_enabled=wandb_enabled, augmentation_params={'augment': True})
