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

object_detection_model = YOLO('yolov8n.pt')

app = FastAPI()

@app.post("/classify")
async def classify(image: UploadFile):
    with Image.open(image.file) as im:
        print(im.mode)
        im = im.convert('RGB') # handle .png files
        print(im.mode)

        # run object detection inference
        object_detection_result = object_detection_model(im, conf=0.1)[0]

        # classify objects in each box detected and draw on image for result
        draw = ImageDraw.Draw(im)
        boxes = object_detection_result.boxes
        for box in boxes:
            # classify object in box
            class_name = object_detection_result.names[int(box.cls)]
            bounding_box = box.xyxy[0].numpy()
            cropped = im.crop(bounding_box)
            
            classification_result = food_classification_model(cropped)[0]
            top3_classes = []
            for i, class_index in enumerate(classification_result.probs.top5[:3]):
                class_name = classification_result.names[class_index]
                class_confidence = classification_result.probs.top5conf[i]
                top3_classes.append({'class_name': class_name, 'class_confidence': class_confidence})

            # draw on image for result
            draw.rectangle(bounding_box)

            info_lines = [f'{info["class_name"]} ({info["class_confidence"] * 100:.2f}%)' for info in top3_classes]
            info_lines.append(f"box.conf: {float(box.conf) * 100:.2f}%")

            font = ImageFont.truetype('arial.ttf', size=10)
            draw.multiline_text(
                bounding_box[:2],
                "\n".join(info_lines),
                font=font,
                fill=(255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0)
            )

        im.save(website_path / 'result.jpg')

    return RedirectResponse('/', status_code=303)

app.mount("/", StaticFiles(directory=website_path, html=True), name="static")

if __name__ == '__main__':
    uvicorn.run("api:app", reload=True)
