from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw
import pandas as pd

from common import cwd, datasets_path

def main():
    model = YOLO('runs/detect/train12/weights/last.pt')
    # image_to_predict_path = datasets_path / 'prannays_edibles_extended' / 'train' / 'dessert' / '2_3.jpg'

    image_to_predict_path = cwd / 'images_to_predict' / 'prannays_edibles' / 'wholesomeyum-Perfect-Grilled-Sirloin-Steak-500x500.jpg'
    img = Image.open(image_to_predict_path)
    results = model(img, conf=0.1)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_name = result.names[int(box.cls)]
            bounding_box = box.xyxy[0].numpy()
            cropped = img.crop(bounding_box)
            # cropped.show()

        result_plotted = result.plot()
        cv2.imshow("result", result_plotted)
        cv2.waitKey()


def test():
    dataset_path = datasets_path / 'UECFOOD100'
    image_no = 11443
    image_class_idx = 17
    test_image_path = dataset_path / f'{image_class_idx}' / f'{image_no}.jpg'
    
    bb_info_path = dataset_path / f'{image_class_idx}' / 'bb_info.txt'
    bb_info_df = pd.read_csv(bb_info_path, delim_whitespace=True)
    
    image_bb = bb_info_df.loc[bb_info_df['img'] == image_no].iloc[0, 1:].to_numpy()
    
    with Image.open(test_image_path) as im:
        draw = ImageDraw.Draw(im)
        draw.rectangle(tuple(image_bb))
        im.show()


"""
Food-related classes:

"39":"bottle",
"40":"wine glass",
"41":"cup",
"42":"fork",
"43":"knife",
"44":"spoon",
"45":"bowl",
"46":"banana",
"47":"apple",
"48":"sandwich",
"49":"orange",
"50":"broccoli",
"51":"carrot",
"52":"hot dog",
"53":"pizza",
"54":"donut",
"55":"cake",

"60":"dining table",
"""


if __name__ == '__main__':
    main()
    # test()
