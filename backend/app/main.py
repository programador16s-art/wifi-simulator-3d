from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .engine import (
    SimulationError,
    extract_cut,
    get_cached_simulation,
    optimize_router,
    simulate,
)
from .materials import MATERIALS
from .schemas import CutAxis, Frequency, SimulationRequest
from .storage import StorageError, mysql_is_configured, save_simulation

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="WiFiMap 3D Simulation API",
    description="Motor de propagación Wi-Fi 3D con pérdidas por espacio libre y obstáculos.",
    version="0.1.0",
)

default_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://programador16s-art.github.io",
]
configured_origins = os.getenv("CORS_ORIGINS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins.split(",") if configured_origins else default_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(SimulationError)
async def simulation_error_handler(_, error: SimulationError):
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/simulador", include_in_schema=False)
@app.get("/simulador.html", include_in_schema=False)
def simulator_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "simulador.html")


@app.get("/api-config.js", include_in_schema=False)
def simulator_api_config() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "api-config.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": "FSPL + 3D AABB obstacle intersections",
        "database_configured": mysql_is_configured(),
        "coordinate_system": "X/Y floor plane in metres; Z height in metres",
    }


@app.get("/api/materiales")
def list_materials() -> dict:
    return {
        "source_note": "All default attenuation values are orientative estimates, not experimental measurements.",
        "materials": [
            {
                "id": material_id,
                "name": item.name,
                "attenuation_2_4_ghz_db": item.loss_24_ghz_db,
                "attenuation_5_ghz_db": item.loss_5_ghz_db,
                "description": item.description,
                "source": item.source,
                "configurable": True,
            }
            for material_id, item in MATERIALS.items()
        ],
    }


@app.post("/api/simulacion/calcular")
def calculate(request: SimulationRequest) -> dict:
    try:
        result, response = simulate(request)
        try:
            save_simulation(result.simulation_id, request)
        except StorageError as error:
            logger.exception("No se pudo guardar la configuración de simulación.")
            raise HTTPException(
                status_code=503,
                detail="La simulación se calculó, pero no se pudo guardar en MySQL.",
            ) from error
        return response
    except SimulationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/simulacion/comparar")
def compare(request: SimulationRequest) -> dict:
    try:
        comparisons = {}
        for frequency in (Frequency.GHZ_2_4, Frequency.GHZ_5):
            _, result = simulate(request, frequency)
            comparisons[frequency.value] = result
        return {
            "same_geometry_and_transmitter": True,
            "comparison": comparisons,
        }
    except SimulationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/simulacion/optimizar")
def optimize(request: SimulationRequest) -> dict:
    try:
        return optimize_router(request)
    except SimulationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/simulacion/cortes")
def get_cut(
    axis: CutAxis,
    simulation_id: str = Query(min_length=1, max_length=80),
    coordinate_m: float = Query(ge=0, le=100),
) -> dict:
    result = get_cached_simulation(simulation_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Simulación no encontrada en memoria. Vuelve a ejecutar el cálculo.",
        )
    try:
        return extract_cut(result, axis.value, coordinate_m)
    except SimulationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
