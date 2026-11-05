from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

import numpy as np
from scipy.optimize import differential_evolution

from .materials import MATERIALS
from .physics import add_powers_dbm, distance_3d_m, fspl_db, thermal_noise_dbm
from .schemas import AccessPoint, Frequency, Obstacle, Point3D, SimulationRequest

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
    sinr_db: np.ndarray
    serving_ap_index: np.ndarray
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


def _inside_obstacle_box(point: np.ndarray, obstacle: Obstacle) -> bool:
    center = np.asarray(obstacle.center.as_tuple())
    half_size = np.asarray(
        [obstacle.dimensions.x, obstacle.dimensions.y, obstacle.dimensions.z], dtype=float
    ) / 2
    return bool(np.all(np.abs(point - center) < half_size))


def _inside_door_opening(point: np.ndarray, obstacle: Obstacle) -> bool:
    opening = obstacle.opening
    if obstacle.type != "wall" or opening is None or opening.type != "door":
        return False
    center = np.asarray(obstacle.center.as_tuple(), dtype=float)
    dimensions = np.asarray(
        [obstacle.dimensions.x, obstacle.dimensions.y, obstacle.dimensions.z], dtype=float
    )
    long_axis = int(np.argmax(dimensions[:2]))
    opening_center = center[long_axis] + opening.offset_m
    bottom = center[2] - dimensions[2] / 2 + opening.sill_height_m
    return bool(
        abs(point[long_axis] - opening_center) < opening.width_m / 2
        and bottom < point[2] < bottom + opening.height_m
    )


def _inside_obstacle(point: np.ndarray, obstacle: Obstacle) -> bool:
    return _inside_obstacle_box(point, obstacle) and not _inside_door_opening(point, obstacle)


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


def _segment_crosses_wall_opening(
    source: np.ndarray, points: np.ndarray, obstacle: Obstacle
) -> np.ndarray:
    opening = obstacle.opening
    if obstacle.type != "wall" or opening is None:
        return np.zeros(len(points), dtype=bool)
    center = np.asarray(obstacle.center.as_tuple(), dtype=float)
    dimensions = np.asarray(
        [obstacle.dimensions.x, obstacle.dimensions.y, obstacle.dimensions.z], dtype=float
    )
    thin_axis = int(np.argmin(dimensions[:2]))
    long_axis = 1 - thin_axis
    direction = points - source
    normal_direction = direction[:, thin_axis]
    moving_across_wall = np.abs(normal_direction) >= 1e-12
    safe_direction = np.where(moving_across_wall, normal_direction, 1.0)
    t = (center[thin_axis] - source[thin_axis]) / safe_direction
    crossing = source + direction * t[:, None]
    opening_center = center[long_axis] + opening.offset_m
    opening_bottom = center[2] - dimensions[2] / 2 + opening.sill_height_m
    return (
        moving_across_wall
        & (t >= 0.0)
        & (t <= 1.0)
        & (np.abs(crossing[:, long_axis] - opening_center) < opening.width_m / 2)
        & (crossing[:, 2] > opening_bottom)
        & (crossing[:, 2] < opening_bottom + opening.height_m)
    )


def _coverage_classes(rssi_dbm: np.ndarray, request: SimulationRequest) -> np.ndarray:
    t = request.thresholds
    return np.digitize(
        rssi_dbm,
        [t.dead_below_dbm, t.weak_below_dbm, t.fair_below_dbm, t.good_below_dbm],
    ).astype(np.uint8)


def _evaluate_single_source(
    request: SimulationRequest,
    points: np.ndarray,
    frequency: Frequency | None = None,
    include_obstacle_paths: bool = False,
) -> dict[str, np.ndarray]:
    frequency = frequency or request.frequency
    source = np.asarray(request.router.as_tuple(), dtype=float)
    if any(_inside_obstacle(source, obstacle) for obstacle in request.obstacles):
        raise SimulationError("El punto de acceso no puede ubicarse dentro de un obstáculo sólido.")

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
            opening_crossing = _segment_crosses_wall_opening(source, chunk, obstacle)
            window_crossing = opening_crossing & (
                obstacle.opening is not None and obstacle.opening.type == "window"
            )
            door_crossing = opening_crossing & (
                obstacle.opening is not None and obstacle.opening.type == "door"
            )
            obstacle_hit = hit & ~door_crossing
            material_hit = hit & ~opening_crossing
            obstruction_db += material_hit * _material_loss(obstacle, frequency)
            if obstacle.opening is not None and obstacle.opening.type == "window":
                obstruction_db += window_crossing * MATERIALS["glass"].loss_for(frequency.ghz)
            obstacle_counts[start:end] += obstacle_hit
            if include_obstacle_paths:
                voxel_indices = np.flatnonzero(obstacle_hit)
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
            inside_box = np.all(np.abs(chunk - center) < half_size, axis=1)
            if obstacle.opening is not None and obstacle.opening.type == "door":
                opening = obstacle.opening
                long_axis = int(np.argmax([obstacle.dimensions.x, obstacle.dimensions.y]))
                opening_center = center[long_axis] + opening.offset_m
                opening_bottom = center[2] - half_size[2] + opening.sill_height_m
                in_doorway = (
                    (np.abs(chunk[:, long_axis] - opening_center) < opening.width_m / 2)
                    & (chunk[:, 2] > opening_bottom)
                    & (chunk[:, 2] < opening_bottom + opening.height_m)
                )
                blocked |= inside_box & ~in_doorway
            else:
                blocked |= inside_box
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


def _channel_overlap_weight(first: AccessPoint, second: AccessPoint) -> float:
    if first.frequency != second.frequency:
        return 0.0
    if first.frequency == Frequency.GHZ_2_4:
        return max(0.0, 1.0 - abs(first.channel - second.channel) / 5.0)
    return 1.0 if first.channel == second.channel else 0.0


def _access_points_for_frequency(
    request: SimulationRequest, frequency: Frequency
) -> list[AccessPoint]:
    selected = [
        access_point
        for access_point in request.active_access_points
        if access_point.frequency is None or access_point.frequency == frequency
    ]
    if not selected:
        raise SimulationError(
            f"No hay puntos de acceso configurados para la banda de {frequency.ghz:g} GHz."
        )
    default_channel = 1 if frequency == Frequency.GHZ_2_4 else 36
    return [
        access_point.model_copy(
            update={
                "frequency": frequency,
                "channel": access_point.channel
                if access_point.frequency is not None
                else default_channel,
            }
        )
        for access_point in selected
    ]


def _sinr_by_access_point(
    request: SimulationRequest,
    access_points: list[AccessPoint],
    rssi_by_ap: np.ndarray,
) -> np.ndarray:
    noise_dbm = thermal_noise_dbm(request.bandwidth_hz, request.noise_figure_db)
    noise_mw = 10.0 ** (noise_dbm / 10.0)
    if request.interference_dbm is not None:
        noise_mw += 10.0 ** (request.interference_dbm / 10.0)
    signal_mw = np.power(10.0, np.nan_to_num(rssi_by_ap, nan=-300.0) / 10.0)
    interference_mw = np.zeros_like(signal_mw)
    for receiver_index, access_point in enumerate(access_points):
        for interferer_index, interferer in enumerate(access_points):
            if receiver_index == interferer_index:
                continue
            overlap = _channel_overlap_weight(access_point, interferer)
            if overlap:
                interference_mw[receiver_index] += signal_mw[interferer_index] * overlap
    sinr_by_ap = 10.0 * np.log10(signal_mw / (noise_mw + interference_mw))
    return np.where(np.isfinite(rssi_by_ap), sinr_by_ap, -np.inf)


def evaluate_points(
    request: SimulationRequest,
    points: np.ndarray,
    frequency: Frequency | None = None,
    include_obstacle_paths: bool = False,
) -> dict[str, np.ndarray]:
    active_frequency = frequency or request.frequency
    access_points = _access_points_for_frequency(request, active_frequency)
    per_access_point = []
    for access_point in access_points:
        single_request = request.model_copy(
            update={
                "router": access_point.position,
                "transmit_power_dbm": access_point.transmit_power_dbm,
                "access_points": None,
            }
        )
        per_access_point.append(
            _evaluate_single_source(
                single_request,
                points,
                active_frequency,
                include_obstacle_paths=include_obstacle_paths,
            )
        )

    rssi_by_ap = np.stack([fields["rssi_dbm"] for fields in per_access_point])
    sinr_by_ap = _sinr_by_access_point(request, access_points, rssi_by_ap)
    winning_ap = np.argmax(sinr_by_ap, axis=0)
    valid = per_access_point[0]["valid"]

    def select_field(name: str) -> np.ndarray:
        stacked = np.stack([fields[name] for fields in per_access_point])
        selected = np.take_along_axis(stacked, winning_ap[None, :], axis=0)[0]
        return selected

    rssi = select_field("rssi_dbm")
    rssi[~valid] = np.nan
    sinr = np.take_along_axis(sinr_by_ap, winning_ap[None, :], axis=0)[0]
    sinr[~valid] = np.nan
    selected_ap_index = np.where(valid, winning_ap, -1).astype(np.int16)
    crossings: list[dict[str, int | str]] = []
    if include_obstacle_paths:
        for ap_index, fields in enumerate(per_access_point):
            crossings.extend(
                crossing
                for crossing in fields["obstacle_crossings"]
                if winning_ap[crossing["voxel_flat_index_zyx"]] == ap_index
            )
        if len(crossings) > MAX_OBSTACLE_CROSSINGS:
            raise SimulationError(
                "El detalle de intersecciones excede el límite de respuesta. "
                "Aumenta voxel_size_m o desactiva include_obstacle_paths."
            )

    return {
        "rssi_dbm": rssi,
        "snr_db": select_field("snr_db"),
        "distance_m": select_field("distance_m"),
        "path_loss_db": select_field("path_loss_db"),
        "obstacle_loss_db": select_field("obstacle_loss_db"),
        "obstacle_count": select_field("obstacle_count"),
        "obstacle_crossings": crossings,
        "valid": valid,
        "coverage_class": _coverage_classes(np.nan_to_num(rssi, nan=-200.0), request),
        "sinr_db": sinr,
        "serving_ap_index": selected_ap_index,
    }


def _statistics(
    fields: dict[str, np.ndarray],
    request: SimulationRequest,
    access_points: list | None = None,
) -> dict:
    valid = fields["valid"]
    if not np.any(valid):
        raise SimulationError("La geometría no contiene ningún punto receptor válido.")
    rssi = fields["rssi_dbm"][valid]
    snr = fields["snr_db"][valid]
    sinr = fields["sinr_db"][valid]
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
        "average_sinr_db": round(float(np.mean(sinr)), 2),
        "useful_coverage_percent": round(float(np.mean(rssi >= t.weak_below_dbm) * 100), 2),
        "dead_zone_percent": round(float(np.mean(classes == 0) * 100), 2),
        "coverage_classes_percent": {
            "dead": round(float(np.mean(classes == 0) * 100), 2),
            "weak": round(float(np.mean(classes == 1) * 100), 2),
            "fair": round(float(np.mean(classes == 2) * 100), 2),
            "good": round(float(np.mean(classes == 3) * 100), 2),
            "excellent": round(float(np.mean(classes == 4) * 100), 2),
        },
        "serving_ap_voxels": {
            access_point.id: int(np.count_nonzero(fields["serving_ap_index"][valid] == index))
            for index, access_point in enumerate(access_points or request.active_access_points)
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
    active_access_points = _access_points_for_frequency(request, active_frequency)
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
        sinr_db=fields["sinr_db"].reshape(shape),
        serving_ap_index=fields["serving_ap_index"].reshape(shape),
        valid=fields["valid"].reshape(shape),
        obstacle_crossings=fields["obstacle_crossings"],
        statistics=_statistics(fields, request, active_access_points),
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
        "access_points": [
            {
                "index": index,
                "id": access_point.id,
                "position_m": access_point.position.model_dump(),
                "transmit_power_dbm": access_point.transmit_power_dbm,
                "frequency_ghz": access_point.frequency.ghz,
                "channel": access_point.channel,
            }
            for index, access_point in enumerate(active_access_points)
        ],
        "model": {
            "name": "FSPL + atenuación por intersección AABB",
            "obstacle_loss": "Pérdida estimada por objeto atravesado; no depende del espesor.",
            "reflections": "No modeladas.",
            "empirical_validation": False,
            "matrix_order": "[z, y, x]",
            "multiple_access_points": "Each receiver is assigned to the AP with the highest estimated SINR. Same/overlapping channel interference is estimated from received power; adjacent-channel weights are simplified.",
        },
    }
    if request.include_grid:
        response["fields"] = {
            "rssi_dbm_zyx": _json_matrix(result.rssi_dbm, shape, -200),
            "snr_db_zyx": _json_matrix(result.snr_db, shape, -200),
            "sinr_db_zyx": _json_matrix(result.sinr_db, shape, -200),
            "distance_m_zyx": _json_matrix(result.distance_m, shape, 0),
            "path_loss_db_zyx": _json_matrix(result.path_loss_db, shape, 0),
            "obstacle_loss_db_zyx": _json_matrix(result.obstacle_loss_db, shape, 0),
            "obstacle_count_zyx": result.obstacle_count.tolist(),
            "coverage_class_zyx": result.coverage_class.tolist(),
            "valid_zyx": result.valid.tolist(),
            "serving_ap_index_zyx": result.serving_ap_index.tolist(),
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
        "sinr_db": np.nan_to_num(result.sinr_db[tuple(key)], nan=-200).tolist(),
        "obstacle_loss_db": result.obstacle_loss_db[tuple(key)].tolist(),
        "obstacle_count": result.obstacle_count[tuple(key)].tolist(),
        "coverage_class": result.coverage_class[tuple(key)].tolist(),
        "valid": result.valid[tuple(key)].tolist(),
        "serving_ap_index": result.serving_ap_index[tuple(key)].tolist(),
    }


def optimize_channels(request: SimulationRequest) -> dict:
    frequency = request.frequency
    access_points = _access_points_for_frequency(request, frequency)
    if len(access_points) < 2:
        raise SimulationError("Se necesitan al menos dos AP en la banda seleccionada para optimizar canales.")

    axes = grid_axes(
        (request.dimensions.x, request.dimensions.y, request.dimensions.z),
        max(request.voxel_size_m, 1.0),
    )
    points, _ = _points_from_axes(axes)
    if len(points) > 1_000:
        points = points[:: int(np.ceil(len(points) / 1_000))]

    per_access_point = []
    for access_point in access_points:
        single_request = request.model_copy(
            update={
                "router": access_point.position,
                "transmit_power_dbm": access_point.transmit_power_dbm,
                "access_points": None,
            }
        )
        per_access_point.append(_evaluate_single_source(single_request, points, frequency))
    rssi_by_ap = np.stack([fields["rssi_dbm"] for fields in per_access_point])
    valid = per_access_point[0]["valid"]
    if not np.any(valid):
        raise SimulationError("La geometría no contiene ningún punto receptor válido.")

    candidates = (
        list(range(1, 14))
        if frequency == Frequency.GHZ_2_4
        else [36, 40, 44, 48, 149, 153, 157, 161, 165]
    )

    def score(plan: list[AccessPoint]) -> tuple[float, np.ndarray]:
        sinr_by_ap = _sinr_by_access_point(request, plan, rssi_by_ap)
        best_sinr = np.max(sinr_by_ap, axis=0)[valid]
        objective = 0.7 * float(np.percentile(best_sinr, 10)) + 0.3 * float(np.mean(best_sinr))
        return objective, best_sinr

    initial_plan = access_points
    initial_objective, initial_sinr = score(initial_plan)
    recommended_plan = list(initial_plan)
    current_objective = initial_objective
    passes = 0
    for _ in range(3):
        changed = False
        for index, access_point in enumerate(recommended_plan):
            best_plan = recommended_plan
            best_objective = current_objective
            for channel in candidates:
                if channel == access_point.channel:
                    continue
                candidate_plan = list(recommended_plan)
                candidate_plan[index] = access_point.model_copy(update={"channel": channel})
                candidate_objective, _ = score(candidate_plan)
                if candidate_objective > best_objective + 1e-9:
                    best_plan = candidate_plan
                    best_objective = candidate_objective
            if best_objective > current_objective + 1e-9:
                recommended_plan = best_plan
                current_objective = best_objective
                changed = True
        passes += 1
        if not changed:
            break

    _, recommended_sinr = score(recommended_plan)
    return {
        "frequency_ghz": frequency.ghz,
        "objective": "Maximiza 70% el percentil 10 de SINR y 30% el SINR promedio de la asociación estimada.",
        "evaluation_voxels": int(valid.sum()),
        "iterations": passes,
        "before": {
            "average_sinr_db": round(float(np.mean(initial_sinr)), 2),
            "minimum_sinr_db": round(float(np.min(initial_sinr)), 2),
            "percentile_10_sinr_db": round(float(np.percentile(initial_sinr, 10)), 2),
        },
        "after": {
            "average_sinr_db": round(float(np.mean(recommended_sinr)), 2),
            "minimum_sinr_db": round(float(np.min(recommended_sinr)), 2),
            "percentile_10_sinr_db": round(float(np.percentile(recommended_sinr, 10)), 2),
        },
        "current_channels": [
            {"id": access_point.id, "channel": access_point.channel}
            for access_point in initial_plan
        ],
        "recommended_channels": [
            {"id": access_point.id, "channel": access_point.channel}
            for access_point in recommended_plan
        ],
        "limitations": "Búsqueda heurística local con canales candidatos y SINR aproximado; no garantiza un óptimo global ni modela airtime, tráfico ni interferencia real de equipos.",
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

    def request_with_primary_position(position: np.ndarray) -> SimulationRequest:
        candidate_tuple = tuple(float(value) for value in position)
        candidate_position = Point3D(
            x=candidate_tuple[0], y=candidate_tuple[1], z=candidate_tuple[2]
        )
        access_points = request.access_points
        if access_points is not None:
            active_access_points = _access_points_for_frequency(request, request.frequency)
            primary_id = active_access_points[0].id
            access_points = [
                access_point.model_copy(update={"position": candidate_position})
                if access_point.id == primary_id
                else access_point
                for access_point in access_points
            ]
        return request.model_copy(
            update={"router": candidate_position, "access_points": access_points}
        )

    def objective(candidate: np.ndarray) -> float:
        candidate_tuple = tuple(float(value) for value in candidate)
        if any(_inside_obstacle(np.asarray(candidate_tuple), item) for item in request.obstacles):
            return 10_000.0
        candidate_request = request_with_primary_position(np.asarray(candidate_tuple))
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
    after_request = request_with_primary_position(recommended)
    after_fields = evaluate_points(after_request, points)
    active_access_points = _access_points_for_frequency(request, request.frequency)
    before_stats = _statistics(before_fields, request, active_access_points)
    after_stats = _statistics(after_fields, request, active_access_points)
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
