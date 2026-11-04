from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    loss_24_ghz_db: float
    loss_5_ghz_db: float
    description: str
    source: str = "Estimación orientativa configurable; no es una medición experimental."

    def loss_for(self, frequency_ghz: float) -> float:
        return self.loss_24_ghz_db if frequency_ghz < 3.0 else self.loss_5_ghz_db


MATERIALS: dict[str, Material] = {
    "concrete": Material("Concreto", 12.0, 18.0, "Muro sólido de concreto."),
    "reinforced_concrete": Material(
        "Concreto armado", 18.0, 25.0, "Concreto con refuerzo metálico."
    ),
    "platabanda": Material(
        "Platabanda", 15.0, 22.0, "Losa o cubierta estructural; valor orientativo."
    ),
    "zinc": Material("Lámina de zinc", 20.0, 28.0, "Lámina metálica delgada."),
    "glass": Material("Vidrio", 3.0, 5.0, "Vidrio común; no modela recubrimientos."),
    "wood": Material("Madera", 4.0, 6.0, "Panel o puerta de madera."),
    "drywall": Material("Drywall / yeso", 3.0, 5.0, "Tabique liviano."),
    "brick": Material("Ladrillo", 8.0, 12.0, "Pared de ladrillo."),
    "metal": Material("Metal", 20.0, 28.0, "Obstáculo metálico."),
}
