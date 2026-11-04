from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

import numpy as np
from scipy.optimize import differential_evolution

from .materials import MATERIALS
from .physics import add_powers_dbm, distance_3d_m, fspl_db, thermal_noise_dbm
from .schemas import Frequency, Obstacle, Point3D, SimulationRequest

MAX_VOXELS = 20_000
MAX_OBSTACLE_CROSSINGS = 100_000
MAX_CACHED_SIMULATIONS = 4
CHUNK_SIZE = 8_192
_cache: OrderedDict[str, "GridResult"] = OrderedDict()
_cache_lock = RLock()


class SimulationError(ValueError):
    pass


@dataclass
class GridResult:
    simulation_id: str
    axes: dict[str, np.ndarray]
    shape: tuple[int, int, int]
    rssi_dbm: np.ndarray
    snr_db: np.ndarray
    distance_m: np.ndarray
    path_loss_db: np.ndarray
    obstacle_loss_db: np.ndarray
    obstacle_count: np.ndarray
    coverage_class: np.ndarray
    valid: np.ndarray
    obstacle_crossings: list[dict[str, int | str]]
    statistics: dict[str, float | int | dict[str, float]]


def grid_axes(dimensions: tuple[float, float, float], spacing_m: float) -> dict[str, np.ndarray]:
    """Return voxel-centre coordinates; arrays are stored in [z, y, x] order."""
    if len(dimensions) != 3 or not np.all(np.isfinite(dimensions)) or any(size <= 0 for size in dimensions):
        raise SimulationError("Las dimensiones X, Y y Z deben ser finitas y positivas.")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise SimulationError("El tamaño del vóxel debe ser finito y positivo.")
    x_size, y_size, z_size = dimensions
    axes = {
        "x": np.arange(spacing_m / 2, x_size, spacing_m, dtype=np.float64),
        "y": np.arange(spacing_m / 2, y_size, spacing_m, dtype=np.float64),
        "z": np.arange(spacing_m / 2, z_size, spacing_m, dtype=np.float64),
    }
    for axis, size in zip(("x", "y", "z"), dimensions):
        if axes[axis].size == 0:
            axes[axis] = np.asarray([size / 2], dtype=np.float64)
    voxel_count = int(np.prod([axes["z"].size, axes["y"].size, axes["x"].size]))
    if voxel_count > MAX_VOXELS:
        raise SimulationError(
            f"La cuadrícula tiene {voxel_count} vóxeles; el máximo por simulación es {MAX_VOXELS}. "
            "Aumenta voxel_size_m."
        )
    return axes


def _points_from_axes(axes: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[int, int, int]]:
    zz, yy, xx = np.meshgrid(axes["z"], axes["y"], axes["x"], indexing="ij")
    shape = xx.shape
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    return points, shape


def _material_loss(obstacle: Obstacle, frequency: Frequency) -> float:
    if frequency == Frequency.GHZ_2_4:
        custom = obstacle.attenuation_2_4_ghz_db
    else:
        custom = obstacle.attenuation_5_ghz_db
    if custom is not None:
        return custom
    return MATERIALS[obstacle.material.value].loss_for(frequency.ghz)


def _inside_obstacle(point: np.ndarray, obstacle: Obstacle) -> bool:
    center = np.asarray(obstacle.center.as_tuple())
    half_size = np.asarray(
        [obstacle.dimensions.x, obstacle.dimensions.y, obstacle.dimensions.z], dtype=float
    ) / 2
    return bool(np.all(np.abs(point - center) < half_size))


def _segment_intersects_boxes(
    source: np.ndarray, points: np.ndarray, obstacle: Obstacle
) -> np.ndarray:
    """Vectorized segment/AABB slab test; each obstacle contributes at most once."""
    center = np.asarray(obstacle.center.as_tuple(), dtype=float)
    half_size = np.asarray(
        [obstacle.dimensions.x, obstacle.dimensions.y, obstacle.dimensions.z], dtype=float
    ) / 2
    lower = center - half_size
    upper = center + half_size
    direction = points - source
    parallel = np.abs(direction) < 1e-12
    outside_parallel = np.any(parallel & ((source < lower) | (source > upper)), axis=1)
    safe_direction = np.where(parallel, 1.0, direction)
    t1 = (lower - source) / safe_direction
    t2 = (upper - source) / safe_direction
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    t_min = np.where(parallel, -np.inf, t_min)
    t_max = np.where(parallel, np.inf, t_max)
    enter = np.max(t_min, axis=1)
    leave = np.min(t_max, axis=1)
    return (~outside_parallel) & (leave >= np.maximum(enter, 0.0)) & (enter <= 1.0)


def _coverage_classes(rssi_dbm: np.ndarray, request: SimulationRequest) -> np.ndarray:
    t = request.thresholds
    return np.digitize(
        rssi_dbm,
        [t.dead_below_dbm, t.weak_below_dbm, t.fair_below_dbm, t.good_below_dbm],
    ).astype(np.uint8)


def evaluate_points(
    request: SimulationRequest,
    points: np.ndarray,
    frequency: Frequency | None = None,
    include_obstacle_paths: bool = False,
) -> dict[str, np.ndarray]:
    frequency = frequency or request.frequency
    source = np.asarray(request.router.as_tuple(), dtype=float)
    if any(_inside_obstacle(source, obstacle) for obstacle in request.obstacles):
        raise SimulationError("El router no puede ubicarse dentro de un obstáculo sólido.")

    count = len(points)
    rssi = np.empty(count, dtype=np.float64)
    snr = np.empty(count, dtype=np.float64)
    distances = np.empty(count, dtype=np.float64)
    path_losses = np.empty(count, dtype=np.float64)
    obstacle_losses = np.zeros(count, dtype=np.float64)
    obstacle_counts = np.zeros(count, dtype=np.uint16)
    valid = np.ones(count, dtype=bool)
    crossings: list[dict[str, int | str]] = []
    eirp_dbm = (
        request.transmit_power_dbm
        + request.transmit_antenna_gain_dbi
        - request.transmit_loss_db
    )
    noise_dbm = thermal_noise_dbm(request.bandwidth_hz, request.noise_figure_db)
    if request.interference_dbm is not None:
        noise_dbm = add_powers_dbm(noise_dbm, request.interference_dbm)

    for start in range(0, count, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, count)
        chunk = points[start:end]
        dist = distance_3d_m(chunk, source)
        loss = fspl_db(dist, frequency.ghz)
        blocked = np.zeros(len(chunk), dtype=bool)
        obstruction_db = np.zeros(len(chunk), dtype=np.float64)
        for obstacle_index, obstacle in enumerate(request.obstacles):
            hit = _segment_intersects_boxes(source, chunk, obstacle)
            obstruction_db += hit * _material_loss(obstacle, frequency)
            obstacle_counts[start:end] += hit
            if include_obstacle_paths:
                voxel_indices = np.flatnonzero(hit)
                if len(crossings) + len(voxel_indices) > MAX_OBSTACLE_CROSSINGS:
                    raise SimulationError(
                        "El detalle de intersecciones excede el límite de respuesta. "
                        "Aumenta voxel_size_m o desactiva include_obstacle_paths."
                    )
                crossings.extend(
                    {
                        "voxel_flat_index_zyx": int(start + voxel_index),
                        "obstacle_index": obstacle_index,
                        "obstacle_id": obstacle.id,
                    }
                    for voxel_index in voxel_indices
                )
            center = np.asarray(obstacle.center.as_tuple(), dtype=float)
            half_size = np.asarray(
                [obstacle.dimensions.x, obstacle.dimensions.y, obstacle.dimensions.z],
                dtype=float,
            ) / 2
            blocked |= np.all(np.abs(chunk - center) < half_size, axis=1)
        rx = (
            eirp_dbm
            - loss
            - obstruction_db
            - request.additional_loss_db
            + request.receive_antenna_gain_dbi
        )
        rssi[start:end] = rx
        snr[start:end] = rx - noise_dbm
        distances[start:end] = dist
        path_losses[start:end] = loss
        obstacle_losses[start:end] = obstruction_db
        valid[start:end] = ~blocked

    rssi[~valid] = np.nan
    snr[~valid] = np.nan
    return {
        "rssi_dbm": rssi,
        "snr_db": snr,
        "distance_m": distances,
        "path_loss_db": path_losses,
        "obstacle_loss_db": obstacle_losses,
        "obstacle_count": obstacle_counts,
        "obstacle_crossings": crossings,
        "valid": valid,
        "coverage_class": _coverage_classes(np.nan_to_num(rssi, nan=-200.0), request),
    }


def _statistics(fields: dict[str, np.ndarray], request: SimulationRequest) -> dict:
    valid = fields["valid"]
    if not np.any(valid):
        raise SimulationError("La geometría no contiene ningún punto receptor válido.")
    rssi = fields["rssi_dbm"][valid]
    snr = fields["snr_db"][valid]
    t = request.thresholds
    classes = fields["coverage_class"][valid]
    total = int(valid.sum())
    return {
        "valid_voxels": total,
        "blocked_voxels": int((~valid).sum()),
        "obstacle_crossings": int(fields["obstacle_count"].sum()),
        "average_rssi_dbm": round(float(np.mean(rssi)), 2),
        "minimum_rssi_dbm": round(float(np.min(rssi)), 2),
        "average_snr_db": round(float(np.mean(snr)), 2),
        "useful_coverage_percent": round(float(np.mean(rssi >= t.weak_below_dbm) * 100), 2),
        "dead_zone_percent": round(float(np.mean(classes == 0) * 100), 2),
        "coverage_classes_percent": {
            "dead": round(float(np.mean(classes == 0) * 100), 2),
            "weak": round(float(np.mean(classes == 1) * 100), 2),
            "fair": round(float(np.mean(classes == 2) * 100), 2),
            "good": round(float(np.mean(classes == 3) * 100), 2),
            "excellent": round(float(np.mean(classes == 4) * 100), 2),
        },
    }


def _json_matrix(array: np.ndarray, shape: tuple[int, int, int], fill: float | int) -> list:
    safe = np.where(np.isfinite(array), array, fill).reshape(shape)
    return safe.tolist()


def simulate(request: SimulationRequest, frequency: Frequency | None = None) -> tuple[GridResult, dict]:
    active_frequency = frequency or request.frequency
    axes = grid_axes(
        (request.dimensions.x, request.dimensions.y, request.dimensions.z),
        request.voxel_size_m,
    )
    points, shape = _points_from_axes(axes)
    fields = evaluate_points(
        request, points, active_frequency, include_obstacle_paths=request.include_obstacle_paths
    )
    result = GridResult(
        simulation_id=str(uuid4()),
        axes=axes,
        shape=shape,
        rssi_dbm=fields["rssi_dbm"].reshape(shape),
        snr_db=fields["snr_db"].reshape(shape),
        distance_m=fields["distance_m"].reshape(shape),
        path_loss_db=fields["path_loss_db"].reshape(shape),
        obstacle_loss_db=fields["obstacle_loss_db"].reshape(shape),
        obstacle_count=fields["obstacle_count"].reshape(shape),
        coverage_class=fields["coverage_class"].reshape(shape),
        valid=fields["valid"].reshape(shape),
        obstacle_crossings=fields["obstacle_crossings"],
        statistics=_statistics(fields, request),
    )
    with _cache_lock:
        _cache[result.simulation_id] = result
        _cache.move_to_end(result.simulation_id)
        while len(_cache) > MAX_CACHED_SIMULATIONS:
            _cache.popitem(last=False)

    response = {
        "simulation_id": result.simulation_id,
        "frequency_ghz": active_frequency.ghz,
        "coordinate_system": "X/Y en planta (m), Z altura (m); matrices [z, y, x].",
        "grid": {
            "shape_zyx": list(shape),
            "voxel_size_m": request.voxel_size_m,
            "x_m": axes["x"].tolist(),
            "y_m": axes["y"].tolist(),
            "z_m": axes["z"].tolist(),
        },
        "statistics": result.statistics,
        "model": {
            "name": "FSPL + atenuación por intersección AABB",
            "obstacle_loss": "Pérdida estimada por objeto atravesado; no depende del espesor.",
            "reflections": "No modeladas.",
            "empirical_validation": False,
            "matrix_order": "[z, y, x]",
        },
    }
    if request.include_grid:
        response["fields"] = {
            "rssi_dbm_zyx": _json_matrix(result.rssi_dbm, shape, -200),
            "snr_db_zyx": _json_matrix(result.snr_db, shape, -200),
            "distance_m_zyx": _json_matrix(result.distance_m, shape, 0),
            "path_loss_db_zyx": _json_matrix(result.path_loss_db, shape, 0),
            "obstacle_loss_db_zyx": _json_matrix(result.obstacle_loss_db, shape, 0),
            "obstacle_count_zyx": result.obstacle_count.tolist(),
            "coverage_class_zyx": result.coverage_class.tolist(),
            "valid_zyx": result.valid.tolist(),
            "obstacle_ids": [
                {"index": index, "id": obstacle.id}
                for index, obstacle in enumerate(request.obstacles)
            ],
        }
        if request.include_obstacle_paths:
            response["fields"]["obstacle_crossings"] = result.obstacle_crossings
    return result, response


def get_cached_simulation(simulation_id: str) -> GridResult | None:
    with _cache_lock:
        result = _cache.get(simulation_id)
        if result is not None:
            _cache.move_to_end(simulation_id)
        return result


def extract_cut(result: GridResult, axis: str, coordinate_m: float) -> dict:
    axis_index = {"x": 2, "y": 1, "z": 0}.get(axis)
    if axis_index is None:
        raise SimulationError("El eje del corte debe ser x, y o z.")
    coordinates = result.axes[axis]
    if coordinate_m < coordinates[0] or coordinate_m > coordinates[-1]:
        raise SimulationError(
            f"La coordenada del corte debe estar entre {coordinates[0]:.3g} y {coordinates[-1]:.3g} m."
        )
    index = int(np.argmin(np.abs(coordinates - coordinate_m)))
    key = [slice(None)] * 3
    key[axis_index] = index
    return {
        "simulation_id": result.simulation_id,
        "axis": axis,
        "requested_coordinate_m": coordinate_m,
        "actual_coordinate_m": float(coordinates[index]),
        "coordinate_axes_m": {
            name: values.tolist() for name, values in result.axes.items() if name != axis
        },
        "matrix_order": "[z, y, x] with requested axis removed",
        "rssi_dbm": np.nan_to_num(result.rssi_dbm[tuple(key)], nan=-200).tolist(),
        "snr_db": np.nan_to_num(result.snr_db[tuple(key)], nan=-200).tolist(),
        "obstacle_loss_db": result.obstacle_loss_db[tuple(key)].tolist(),
        "obstacle_count": result.obstacle_count[tuple(key)].tolist(),
        "coverage_class": result.coverage_class[tuple(key)].tolist(),
        "valid": result.valid[tuple(key)].tolist(),
    }


def optimize_router(request: SimulationRequest) -> dict:
    axes = grid_axes(
        (request.dimensions.x, request.dimensions.y, request.dimensions.z),
        max(request.voxel_size_m, 1.0),
    )
    points, _ = _points_from_axes(axes)
    if len(points) > 1_000:
        points = points[:: int(np.ceil(len(points) / 1_000))]

    z_min = min(0.8, request.dimensions.z / 2)
    z_max = min(2.2, request.dimensions.z)
    margin_x = min(0.25, request.dimensions.x / 2)
    margin_y = min(0.25, request.dimensions.y / 2)
    bounds = [
        (margin_x, max(margin_x, request.dimensions.x - margin_x)),
        (margin_y, max(margin_y, request.dimensions.y - margin_y)),
        (z_min, max(z_min, z_max)),
    ]

    def objective(candidate: np.ndarray) -> float:
        candidate_tuple = tuple(float(value) for value in candidate)
        if any(_inside_obstacle(np.asarray(candidate_tuple), item) for item in request.obstacles):
            return 10_000.0
        candidate_request = request.model_copy(
            update={
                "router": Point3D(
                    x=candidate_tuple[0], y=candidate_tuple[1], z=candidate_tuple[2]
                )
            }
        )
        fields = evaluate_points(candidate_request, points)
        valid = fields["valid"]
        if not np.any(valid):
            return 10_000.0
        rssi = fields["rssi_dbm"][valid]
        threshold = request.thresholds.weak_below_dbm
        coverage = float(np.mean(rssi >= threshold))
        mean_quality = float(np.clip((np.mean(rssi) + 100.0) / 45.0, 0.0, 1.0))
        dead_fraction = float(np.mean(rssi < request.thresholds.dead_below_dbm))
        return -(0.65 * coverage + 0.35 * mean_quality) + 0.2 * dead_fraction

    optimized = differential_evolution(
        objective,
        bounds,
        seed=17,
        maxiter=3,
        popsize=5,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.08,
    )
    original_position = np.asarray(request.router.as_tuple(), dtype=float)
    recommended = np.asarray(optimized.x, dtype=float)
    if objective(recommended) >= 1_000:
        raise SimulationError("No se encontró una ubicación válida para el router.")

    before_fields = evaluate_points(request, points)
    after_request = request.model_copy(
        update={
            "router": Point3D(
                x=float(recommended[0]), y=float(recommended[1]), z=float(recommended[2])
            )
        }
    )
    after_fields = evaluate_points(after_request, points)
    before_stats = _statistics(before_fields, request)
    after_stats = _statistics(after_fields, request)
    return {
        "original_position_m": dict(zip(("x", "y", "z"), original_position.tolist())),
        "recommended_position_m": dict(zip(("x", "y", "z"), recommended.tolist())),
        "evaluation_voxels": len(points),
        "coverage_before_percent": before_stats["useful_coverage_percent"],
        "coverage_after_percent": after_stats["useful_coverage_percent"],
        "coverage_change_percentage_points": round(
            after_stats["useful_coverage_percent"] - before_stats["useful_coverage_percent"], 2
        ),
        "dead_zone_before_percent": before_stats["dead_zone_percent"],
        "dead_zone_after_percent": after_stats["dead_zone_percent"],
        "objective": "Maximiza 65% cobertura >= umbral débil + 35% calidad RSSI normalizada y penaliza zonas muertas.",
        "method": "SciPy differential_evolution; cuadrícula de evaluación reducida hasta 1.000 puntos.",
        "limitations": "Recomendación aproximada para las condiciones y obstáculos proporcionados.",
    }
