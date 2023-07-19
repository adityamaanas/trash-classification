from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from fastai.vision.all import *
import skimage
from urllib.request import urlopen
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)

learn = load_learner("https://github.com/adityamaanas/trash-classification/blob/main/export.pkl")


@app.get("/")
async def root():
    return {"message": "Welcome to the Garbage Classification API!"}


@app.post("/predict")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    pred, idx, prob = learn.predict(PILImage.create(urlopen(image_link)))
    return {"prediction": pred, "probability": float(prob[0])}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app)
