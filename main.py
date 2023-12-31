from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from fastai.vision.all import * # type: ignore
import skimage
from urllib.request import urlopen
import os
import croppingbox

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

learn = load_learner("C:/Users/adipi/Documents/Code/Internship-RAKRIC-0723/trash-classification/export.pkl")


@app.get("/")
async def root():
    return {"message": "Welcome to the Garbage Classification API!"}


@app.post("/predict")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    
    cropped = croppingbox.crop(image_link)
    predictions = []
    # Predict for each cropped image
    for i in range(len(cropped)):
        pred, idx, prob = learn.predict(cropped[i])
        predictions.append({"prediction": pred})
        #predictions.append({"prediction": pred, "probability": float(prob[0])})
        
    return predictions

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app)

#python -m uvicorn main:app --reload