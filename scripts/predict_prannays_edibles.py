from pathlib import Path
from ultralytics import YOLO

def main():
    model_path = Path('runs') / 'classify' / 'train14' / 'weights' / 'best.pt'
    model_absolute_path = model_path.absolute()
    model = YOLO(model_absolute_path)

    image_to_predict_path = Path('images_to_predict') / 'prannays_edibles' / 'steak.jpg'
    results = model(str(image_to_predict_path))
    for result in results:
        for i, contender in enumerate(result.probs.top5):
            print(i, result.names[contender], f"({result.probs.top5conf[i] * 100:.2f}% confidence)")


if __name__ == '__main__':
    main()
