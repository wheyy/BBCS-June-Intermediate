from pathlib import Path
import os

from ultralytics import YOLO
import wandb

from common import project_name, cwd, datasets_path

def main():
    if "WANDB_API_KEY" not in os.environ:
        wandb_api_key = input("Enter wandb API key: ")
        os.environ["WANDB_API_KEY"] = wandb_api_key

    wandb_run_name = input("Enter name of wandb run to download model from: ")

    wandb_downloaded_artifacts_path = cwd / 'wandb_downloaded_artifacts'
    wandb_downloaded_artifacts_path.mkdir(exist_ok=True)

    if wandb_run_name != "reuse":
        api = wandb.Api()
        artifact = api.artifact(f'{project_name}/{wandb_run_name}:v0')
        artifact.download(wandb_downloaded_artifacts_path)

    model_path = wandb_downloaded_artifacts_path / 'best.pt'
    model = YOLO(model_path.absolute())

    dataset_path = datasets_path / 'prannays_edibles_extended'
    metrics = model.val(str(dataset_path))
    print(metrics)


def get_local_model_path():
    model_path = Path('runs') / 'classify' / 'train14' / 'weights' / 'best.pt'
    model_absolute_path = model_path.absolute()
    return model_absolute_path


if __name__ == '__main__':
    main()
