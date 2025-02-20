import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine

from lineworld.core.layerstack import LayerStack
from lineworld.layers import bathymetry, contour, coastlines, grid

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

engine = create_engine("postgresql+psycopg://localhost:5432/lineworld", echo=True)

layerstack = LayerStack(
    [
        bathymetry.Bathymetry("Bathymetry", [0, -12_000], 15, engine),
        contour.Contour("Contour", [0, 9_000], 15, engine),
        coastlines.Coastlines("Coastlines", engine),
        grid.Grid("Grid", engine),
    ]
)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/layers")
async def get_layer_info():
    # return {{"layer_name": k, "z": i} for i, (k, v) in enumerate(layers.items())}
    return layerstack


@app.get("/layer/{layer_name}")
async def get_layer(layer_name: str):
    return layerstack.get(layer_name)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
