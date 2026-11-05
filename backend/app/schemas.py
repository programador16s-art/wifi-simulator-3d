from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class Frequency(str, Enum):
    GHZ_2_4 = "2.4"
    GHZ_5 = "5"

    @property
    def ghz(self) -> float:
        return float(self.value)


class MaterialName(str, Enum):
    CONCRETE = "concrete"
    REINFORCED_CONCRETE = "reinforced_concrete"
    PLATABANDA = "platabanda"
    ZINC = "zinc"
    GLASS = "glass"
    WOOD = "wood"
    DRYWALL = "drywall"
    BRICK = "brick"
    METAL = "metal"


class Point3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: FiniteFloat = Field(ge=0, le=100)
    y: FiniteFloat = Field(ge=0, le=100)
    z: FiniteFloat = Field(ge=0, le=30)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Dimensions3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: FiniteFloat = Field(gt=0, le=100)
    y: FiniteFloat = Field(gt=0, le=100)
    z: FiniteFloat = Field(gt=0, le=30)


class WallOpening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["door", "window"]
    width_m: FiniteFloat = Field(gt=0, le=20)
    height_m: FiniteFloat = Field(gt=0, le=20)
    offset_m: FiniteFloat = Field(default=0, ge=-50, le=50)
    sill_height_m: FiniteFloat = Field(default=0, ge=0, le=20)


class AccessPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    position: Point3D
    transmit_power_dbm: FiniteFloat = Field(default=20.0, ge=-10, le=40)
    frequency: Frequency | None = None
    channel: int | None = Field(default=None, ge=1, le=196)

    @model_validator(mode="after")
    def validate_radio_channel(self) -> "AccessPoint":
        if self.frequency is None:
            if self.channel is not None:
                raise ValueError("El canal requiere una banda de frecuencia explícita.")
            return self
        if self.channel is None:
            raise ValueError("Cada radio debe indicar un canal.")
        if self.frequency == Frequency.GHZ_2_4 and self.channel not in range(1, 14):
            raise ValueError("En 2.4 GHz se admiten canales del 1 al 13.")
        if self.frequency == Frequency.GHZ_5 and self.channel not in {
            36, 40, 44, 48, 149, 153, 157, 161, 165
        }:
            raise ValueError("Canal de 5 GHz no disponible en este modelo.")
        return self


class Obstacle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    type: Literal[
        "box",
        "wall",
        "floor",
        "ceiling",
        "column",
        "furniture",
        "table",
        "sofa",
        "shelf",
        "cabinet",
    ] = "box"
    center: Point3D
    dimensions: Dimensions3D
    material: MaterialName
    opening: WallOpening | None = None
    attenuation_2_4_ghz_db: FiniteFloat | None = Field(default=None, ge=0, le=100)
    attenuation_5_ghz_db: FiniteFloat | None = Field(default=None, ge=0, le=100)


class CoverageThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dead_below_dbm: FiniteFloat = -75.0
    weak_below_dbm: FiniteFloat = -67.0
    fair_below_dbm: FiniteFloat = -60.0
    good_below_dbm: FiniteFloat = -50.0

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "CoverageThresholds":
        values = (
            self.dead_below_dbm,
            self.weak_below_dbm,
            self.fair_below_dbm,
            self.good_below_dbm,
        )
        if tuple(sorted(values)) != values:
            raise ValueError("Los umbrales RSSI deben estar ordenados de menor a mayor.")
        return self


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: Dimensions3D
    router: Point3D
    access_points: list[AccessPoint] | None = Field(default=None, min_length=1, max_length=8)
    frequency: Frequency = Frequency.GHZ_2_4
    transmit_power_dbm: FiniteFloat = Field(default=20.0, ge=-10, le=40)
    transmit_antenna_gain_dbi: FiniteFloat = Field(default=0.0, ge=-10, le=20)
    transmit_loss_db: FiniteFloat = Field(default=0.0, ge=0, le=30)
    receive_antenna_gain_dbi: FiniteFloat = Field(default=0.0, ge=-10, le=20)
    bandwidth_hz: FiniteFloat = Field(default=20_000_000.0, gt=0, le=1_000_000_000)
    noise_figure_db: FiniteFloat = Field(default=7.0, ge=0, le=30)
    interference_dbm: FiniteFloat | None = Field(default=None, ge=-174, le=100)
    additional_loss_db: FiniteFloat = Field(default=0.0, ge=0, le=100)
    voxel_size_m: FiniteFloat = Field(default=0.5, ge=0.1, le=5)
    obstacles: list[Obstacle] = Field(default_factory=list, max_length=200)
    thresholds: CoverageThresholds = Field(default_factory=CoverageThresholds)
    include_grid: bool = True
    include_obstacle_paths: bool = True

    @model_validator(mode="after")
    def validate_room_and_obstacles(self) -> "SimulationRequest":
        d = self.dimensions
        r = self.router
        if r.x > d.x or r.y > d.y or r.z > d.z:
            raise ValueError("El router debe estar dentro de las dimensiones del inmueble.")
        ids = [obstacle.id for obstacle in self.obstacles]
        if len(ids) != len(set(ids)):
            raise ValueError("Los identificadores de obstáculos deben ser únicos.")
        if self.access_points is not None:
            access_point_ids = [access_point.id for access_point in self.access_points]
            if len(access_point_ids) != len(set(access_point_ids)):
                raise ValueError("Los identificadores de puntos de acceso deben ser únicos.")
            for access_point in self.access_points:
                position = access_point.position
                if position.x > d.x or position.y > d.y or position.z > d.z:
                    raise ValueError(
                        f"El punto de acceso '{access_point.id}' debe estar dentro de las dimensiones del inmueble."
                    )
        for obstacle in self.obstacles:
            for axis in ("x", "y", "z"):
                center = getattr(obstacle.center, axis)
                size = getattr(obstacle.dimensions, axis)
                limit = getattr(d, axis)
                if center - size / 2 < 0 or center + size / 2 > limit:
                    raise ValueError(
                        f"El obstáculo '{obstacle.id}' excede el inmueble en el eje {axis.upper()}."
                    )
            if obstacle.opening is not None:
                if obstacle.type != "wall":
                    raise ValueError("Las puertas y ventanas solo pueden pertenecer a un obstáculo de tipo pared.")
                horizontal_dimensions = (obstacle.dimensions.x, obstacle.dimensions.y)
                short_span = min(horizontal_dimensions)
                long_span = max(horizontal_dimensions)
                if short_span >= long_span * 0.25:
                    raise ValueError("La pared con abertura debe ser una caja alargada y delgada.")
                opening = obstacle.opening
                if opening.width_m > long_span:
                    raise ValueError("El ancho de la abertura no puede superar el largo de la pared.")
                if abs(opening.offset_m) + opening.width_m / 2 > long_span / 2:
                    raise ValueError("La abertura debe quedar dentro de los extremos de la pared.")
                if opening.sill_height_m + opening.height_m > obstacle.dimensions.z:
                    raise ValueError("La abertura debe quedar dentro de la altura de la pared.")
                if opening.type == "door" and opening.sill_height_m != 0:
                    raise ValueError("Una puerta debe comenzar al nivel del suelo de la pared.")
        return self

    @property
    def active_access_points(self) -> list[AccessPoint]:
        if self.access_points is not None:
            return self.access_points
        return [
            AccessPoint(
                id="ap-1",
                position=self.router,
                transmit_power_dbm=self.transmit_power_dbm,
            )
        ]


class CutAxis(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"
