from pathlib import Path

from fastapi import FastAPI, UploadFile
import uvicorn
from ultralytics import YOLO
import numpy as np
from PIL import Image

cwd = Path.cwd()
app_path = cwd / 'app'
models_path = app_path / 'models'
website_path = app_path / 'website'

food_classification_model_path = models_path / 'food_classification.pt'

app = FastAPI()

@app.get("/")
async def root():
    return "Nothing to see here"


@app.post("/classify")
async def classify(image: UploadFile):
    model = YOLO(str(food_classification_model_path.absolute()))

    with Image.open(image.file) as image_pil:
        results = model(image_pil)
    
    result = results[0]

    output = {}

    for i, contender in enumerate(result.probs.top5):
        class_name = result.names[contender]
        class_confidence = result.probs.top5conf[i]
        output[class_name] = round(float(class_confidence), 4)

    return output


if __name__ == '__main__':
    uvicorn.run("api:app", reload=True)
