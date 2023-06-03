import logging
import shutil
from pathlib import Path

from torch.utils.data import random_split
from torchvision import datasets

from common import prannays_edibles_path, prannays_edibles_class_name_map

def main():
    split_dataset_path = create_image_classify_data_split_folder(
        prannays_edibles_path,
        prannays_edibles_class_name_map,
        train_split_percentage=0.7,
        recreate=True
    )
    print(split_dataset_path)

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
    dataset = datasets.DatasetFolder(root=not_yet_split_dataset_path, loader=lambda path: path, is_valid_file=lambda x: True)

    test_split_percentage = 1 - train_split_percentage

    train_dataset, test_dataset = random_split(dataset, [train_split_percentage, test_split_percentage])

    def get_class_name(class_index):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        return class_name_map[idx_to_class[class_index]]


    def create_dataset_folder(dataset, dataset_path):
        for i, (image_path_str, image_class_idx) in enumerate(dataset):
            image_class_name = get_class_name(image_class_idx)
            class_path = dataset_path / image_class_name
            if not class_path.exists():
                class_path.mkdir(exist_ok=True)
            
            image_path = Path(image_path_str)
            new_image_path = class_path / f"{image_class_name}_{i}.jpg"
            shutil.copyfile(image_path, new_image_path)


    create_dataset_folder(train_dataset, train_dataset_path)
    create_dataset_folder(test_dataset, test_dataset_path)

    return split_dataset_path


if __name__ == '__main__':
    main()
