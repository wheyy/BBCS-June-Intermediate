from pathlib import Path
import os

from ultralytics import YOLO
import wandb

from common import project_name, cwd

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

    image_to_predict_path = Path('images_to_predict') / 'prannays_edibles' / 'wholesomeyum-Perfect-Grilled-Sirloin-Steak-500x500.jpg'
    results = model(str(image_to_predict_path))
    for result in results:
        for i, contender in enumerate(result.probs.top5):
            print(i, result.names[contender], f"({result.probs.top5conf[i] * 100:.2f}% confidence)")


def get_local_model_path():
    model_path = Path('runs') / 'classify' / 'train14' / 'weights' / 'best.pt'
    model_absolute_path = model_path.absolute()
    return model_absolute_path


if __name__ == '__main__':
    main()
