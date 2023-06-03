from ultralytics import YOLO

def main():
    model = YOLO('runs/detect/train12/weights/last.pt')
    metrics = model.val('datasets\\UECFOOD100_prepared\\UECFOOD100.yaml')
    print(metrics)


if __name__ == '__main__':
    main()
