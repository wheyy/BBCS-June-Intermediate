from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

cwd = Path.cwd()
app_path = cwd / 'app'
models_path = app_path / 'models'
website_path = app_path / 'website'

food_classification_model_path = models_path / 'food_classification.pt'
food_classification_model = YOLO(str(food_classification_model_path.absolute()))

food_classification_model_path = models_path / 'food_object_detection.pt'
food_object_detection_model = YOLO(str(food_classification_model_path.absolute()))

font = ImageFont.truetype('arial.ttf', size=16)
text_kwags = {
    'fill': (255, 255, 255),
    'stroke_width': 4,
    'stroke_fill': (0, 0, 0),
}

def get_scaled_font(img: Image.Image, scale=0.05):
    return ImageFont.truetype('arial.ttf', size=max(int(img.size[1] * scale), 12))


app = FastAPI()

@app.post("/classify")
async def classify(image: UploadFile):
    with Image.open(image.file) as im:
        im = im.convert('RGB') # handle .png files

        # run object detection inference
        object_detection_result = food_object_detection_model(im, conf=0.02, iou=0.3)[0]

        # classify objects in each box detected and draw on image for result
        im_with_boxes_drawn = im.copy()
        main_draw = ImageDraw.Draw(im_with_boxes_drawn)
        cropped_images = []

        boxes = object_detection_result.boxes
        for box in boxes:
            # classify object in box
            class_name = object_detection_result.names[int(box.cls)]
            bounding_box = box.xyxy[0].numpy()
            cropped = im.crop(bounding_box)
            cropped_images.append(cropped)
            
            classification_result = food_classification_model(cropped)[0]
            top3_classes = []
            for i, class_index in enumerate(classification_result.probs.top5[:3]):
                class_name = classification_result.names[class_index]
                class_confidence = classification_result.probs.top5conf[i]
                top3_classes.append({'class_name': class_name, 'class_confidence': class_confidence})

            # draw on image for result
            main_draw.rectangle(bounding_box)

            info_lines = [f'{info["class_name"]} ({info["class_confidence"] * 100:.2f}%)' for info in top3_classes]
            cropped_draw = ImageDraw.Draw(cropped)
            cropped_draw.text(
                (0, 0),
                "\n".join(info_lines),
                font=get_scaled_font(cropped),
                **text_kwags,
            )

            main_draw.text(
                bounding_box[:2],
                f"box.conf: {float(box.conf) * 100:.2f}%",
                font=get_scaled_font(im_with_boxes_drawn, 0.02),
                **text_kwags
            )

        all_images = [im_with_boxes_drawn] + cropped_images
        widths, heights = zip(*(i.size for i in all_images))
        
        max_width = max(widths)
        total_height = sum(heights)

        result_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for each_im in all_images:
            result_im.paste(each_im, (0, y_offset))
            y_offset += each_im.size[1]

        result_im.save(website_path / 'result.jpg')

    return RedirectResponse('/', status_code=303)

app.mount("/", StaticFiles(directory=website_path, html=True), name="static")

if __name__ == '__main__':
    uvicorn.run("api:app", reload=True)
