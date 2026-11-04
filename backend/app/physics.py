from __future__ import annotations

import math

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0
MIN_DISTANCE_M = 0.1
THERMAL_NOISE_DENSITY_DBM_HZ = -174.0


def distance_3d_m(points: np.ndarray, source: tuple[float, float, float]) -> np.ndarray:
    """Euclidean distance in metres; input columns and source are X, Y, Z."""
    coordinates = np.asarray(points, dtype=float)
    origin = np.asarray(source, dtype=float)
    if coordinates.shape[-1:] != (3,) or origin.shape != (3,):
        raise ValueError("Los puntos y el origen deben usar tres coordenadas X, Y y Z.")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(origin)):
        raise ValueError("Las coordenadas deben ser finitas.")
    return np.linalg.norm(coordinates - origin, axis=-1)


def fspl_db(distance_m: np.ndarray | float, frequency_ghz: float) -> np.ndarray:
    """Free-space path loss; distance is converted to km and GHz to MHz."""
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        raise ValueError("La frecuencia debe ser un valor positivo en GHz.")
    distances = np.asarray(distance_m, dtype=float)
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Las distancias deben ser finitas y no negativas.")
    distance_km = np.maximum(distances, MIN_DISTANCE_M) / 1000.0
    frequency_mhz = frequency_ghz * 1000.0
    return 20.0 * np.log10(distance_km) + 20.0 * np.log10(frequency_mhz) + 32.44


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float = 0.0) -> float:
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
        raise ValueError("El ancho de banda debe ser positivo y estar expresado en Hz.")
    if not math.isfinite(noise_figure_db) or noise_figure_db < 0:
        raise ValueError("La figura de ruido no puede ser negativa.")
    return THERMAL_NOISE_DENSITY_DBM_HZ + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def add_powers_dbm(*powers_dbm: float) -> float:
    """Add independent powers in linear milliwatts, returning dBm."""
    if not powers_dbm:
        raise ValueError("Se requiere al menos una potencia.")
    powers = np.asarray(powers_dbm, dtype=float)
    if not np.all(np.isfinite(powers)):
        raise ValueError("Las potencias deben ser finitas.")
    return float(10.0 * np.log10(np.sum(np.power(10.0, powers / 10.0))))
