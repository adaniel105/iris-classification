from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import numpy as np
import pickle

static_folder = Path(__file__).parent.resolve(strict=True) / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(static_folder)), name="static")


template_folder = Path(__file__).parent.resolve(strict=True) / "templates"
templates = Jinja2Templates(directory=str(template_folder))

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, prediction_str: str = " "):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_str": prediction_str},
    )


@app.post("/predict")
def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    preds = model.predict(input_features)

    species_map = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
    predicted_species = species_map.get(preds[0])
    return templates.TemplateResponse(
        "index.html", {"request": request, "prediction_str": predicted_species}
    )
