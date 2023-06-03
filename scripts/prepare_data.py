import logging
import shutil
from pathlib import Path

from common import prannays_edibles_class_name_map, food11_class_name_map, prannays_edibles_dataset_path, food11_dataset_path

def main():
    dataset_path = prepare_data(
        prannays_edibles_dataset_path,
        prannays_edibles_class_name_map,
        food11_dataset_path,
        food11_class_name_map,
    )
    print(dataset_path)

def prepare_data(prannays_dataset_path: Path, prannays_edibles_class_name_map: dict, food11_dataset_path: Path, food11_class_name_map):
    logging.info("prepare_data running")
    extended_dataset_path = prannays_dataset_path.parent / f"{prannays_dataset_path.name}_extended"
    if extended_dataset_path.exists():
        return extended_dataset_path

    extended_dataset_path.mkdir(exist_ok=True)

    train_dataset_path = extended_dataset_path / "train"
    train_dataset_path.mkdir(exist_ok=True)

    test_dataset_path = extended_dataset_path / "test"
    test_dataset_path.mkdir(exist_ok=True)

    val_dataset_path = extended_dataset_path / "val"
    val_dataset_path.mkdir(exist_ok=True)

    # copy prannays edibles to train folder
    for class_path in prannays_dataset_path.iterdir():
        class_name = prannays_edibles_class_name_map[class_path.name]
        new_class_path = train_dataset_path / class_name
        shutil.copytree(class_path, new_class_path)
    
    # copy food11 evaluation folder to test folder
    for class_path in (food11_dataset_path / 'evaluation').iterdir():
        class_name = food11_class_name_map[class_path.name]
        new_class_path = test_dataset_path / class_name
        shutil.copytree(class_path, new_class_path)

    # copy food11 validation folder to val folder
    for class_path in (food11_dataset_path / 'validation').iterdir():
        class_name = food11_class_name_map[class_path.name]
        new_class_path = val_dataset_path / class_name
        shutil.copytree(class_path, new_class_path)

    return extended_dataset_path


if __name__ == '__main__':
    main()
