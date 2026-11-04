from __future__ import annotations

import json
import os

from .schemas import SimulationRequest


class StorageError(RuntimeError):
    pass


def mysql_is_configured() -> bool:
    keys = ("MYSQL_HOST", "MYSQL_USER", "MYSQL_DATABASE")
    return all(os.getenv(key) for key in keys)


def save_simulation(simulation_id: str, request: SimulationRequest) -> None:
    """Persist only reproducible configuration metadata; never store the voxel arrays."""
    required = ("MYSQL_HOST", "MYSQL_USER", "MYSQL_DATABASE")
    configured = [bool(os.getenv(key)) for key in required]
    if not any(configured):
        return
    if not all(configured):
        raise StorageError("La configuración MySQL está incompleta; revisa las variables MYSQL_*.")

    import mysql.connector
    from mysql.connector import Error

    connection = None
    try:
        connection = mysql.connector.connect(
            host=os.environ["MYSQL_HOST"],
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.environ["MYSQL_USER"],
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.environ["MYSQL_DATABASE"],
            connection_timeout=5,
        )
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO simulaciones
                (id_simulacion, frecuencia_ghz, potencia_transmision_dbm,
                 ancho_banda_hz, dimensiones_x_m, dimensiones_y_m,
                 dimensiones_z_m, resolucion_espacial_m)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                simulation_id,
                request.frequency.ghz,
                request.transmit_power_dbm,
                request.bandwidth_hz,
                request.dimensions.x,
                request.dimensions.y,
                request.dimensions.z,
                request.voxel_size_m,
            ),
        )
        cursor.execute(
            """
            INSERT INTO configuraciones
                (id_simulacion, router_x_m, router_y_m, router_z_m, parametros_json)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                simulation_id,
                request.router.x,
                request.router.y,
                request.router.z,
                json.dumps(
                    {
                        "transmit_antenna_gain_dbi": request.transmit_antenna_gain_dbi,
                        "transmit_loss_db": request.transmit_loss_db,
                        "receive_antenna_gain_dbi": request.receive_antenna_gain_dbi,
                        "noise_figure_db": request.noise_figure_db,
                        "interference_dbm": request.interference_dbm,
                        "additional_loss_db": request.additional_loss_db,
                        "thresholds": request.thresholds.model_dump(),
                    }
                ),
            ),
        )
        for obstacle in request.obstacles:
            cursor.execute(
                """
                INSERT INTO obstaculos
                    (id_simulacion, id_obstaculo, tipo, material,
                     posicion_x_m, posicion_y_m, posicion_z_m, dimensiones_json,
                     parametros_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    simulation_id,
                    obstacle.id,
                    obstacle.type,
                    obstacle.material.value,
                    obstacle.center.x,
                    obstacle.center.y,
                    obstacle.center.z,
                    json.dumps(obstacle.dimensions.model_dump()),
                    json.dumps(
                        {
                            "attenuation_2_4_ghz_db": obstacle.attenuation_2_4_ghz_db,
                            "attenuation_5_ghz_db": obstacle.attenuation_5_ghz_db,
                        }
                    ),
                ),
            )
        connection.commit()
        cursor.close()
    except Error as error:
        if connection is not None:
            connection.rollback()
        raise StorageError("MySQL no pudo guardar los metadatos de la simulación.") from error
    finally:
        if connection is not None and connection.is_connected():
            connection.close()
