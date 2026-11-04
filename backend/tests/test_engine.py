import numpy as np
import pytest
import sympy as sp

from backend.app.equations import distance_km, frequency_mhz, fspl_db_expression
from backend.app.engine import (
    SimulationError,
    _segment_intersects_boxes,
    evaluate_points,
    grid_axes,
    optimize_router,
    simulate,
)
from backend.app.materials import MATERIALS
from backend.app.physics import (
    add_powers_dbm,
    distance_3d_m,
    fspl_db,
    thermal_noise_dbm,
)
from backend.app.schemas import (
    Dimensions3D,
    Frequency,
    Obstacle,
    Point3D,
    SimulationRequest,
)


def make_request(**overrides) -> SimulationRequest:
    values = {
        "dimensions": {"x": 4, "y": 4, "z": 2.4},
        "router": {"x": 0.5, "y": 0.5, "z": 1.2},
        "voxel_size_m": 1,
    }
    values.update(overrides)
    return SimulationRequest.model_validate(values)


def box_obstacle(material: str, obstacle_id: str = "wall") -> Obstacle:
    return Obstacle.model_validate(
        {
            "id": obstacle_id,
            "type": "wall",
            "center": {"x": 2, "y": 2, "z": 1.2},
            "dimensions": {"x": 0.15, "y": 3.8, "z": 2.4},
            "material": material,
        }
    )


def test_distance_uses_all_three_dimensions():
    actual = distance_3d_m(np.array([[3.0, 4.0, 12.0]]), (0.0, 0.0, 0.0))
    assert actual[0] == pytest.approx(13.0)


def test_fspl_increases_with_distance_and_uses_near_field_clamp():
    result = fspl_db(np.array([1.0, 10.0]), 2.4)
    assert result[1] > result[0]
    assert fspl_db(0.0, 2.4) == pytest.approx(fspl_db(0.1, 2.4))
    with pytest.raises(ValueError, match="no negativas"):
        fspl_db(-1, 2.4)
    with pytest.raises(ValueError, match="finitas"):
        distance_3d_m(np.array([[np.inf, 0, 0]]), (0, 0, 0))


def test_segment_box_intersection_deduplicates_geometry_by_obstacle():
    obstacle = box_obstacle("brick")
    points = np.array([[3.5, 0.5, 1.2], [1.0, 3.5, 1.2]])
    hits = _segment_intersects_boxes(np.array([0.5, 0.5, 1.2]), points, obstacle)
    assert hits.tolist() == [True, False]


@pytest.mark.parametrize(
    ("material", "expected"),
    [("brick", 8.0), ("zinc", 20.0), ("reinforced_concrete", 18.0)],
)
def test_obstacle_loss_is_applied_once_per_box(material, expected):
    request = make_request(obstacles=[box_obstacle(material)])
    fields = evaluate_points(
        request, np.array([[3.5, 0.5, 1.2]]), include_obstacle_paths=True
    )
    assert fields["obstacle_loss_db"][0] == pytest.approx(expected)
    assert fields["obstacle_count"][0] == 1
    assert fields["obstacle_crossings"] == [
        {"voxel_flat_index_zyx": 0, "obstacle_index": 0, "obstacle_id": "wall"}
    ]


def test_custom_material_loss_overrides_orientative_default():
    obstacle = box_obstacle("brick").model_copy(
        update={"attenuation_2_4_ghz_db": 13.5}
    )
    request = make_request(obstacles=[obstacle])
    fields = evaluate_points(request, np.array([[3.5, 0.5, 1.2]]))
    assert fields["obstacle_loss_db"][0] == pytest.approx(13.5)
    assert "no es una medición experimental" in MATERIALS["brick"].source


@pytest.mark.parametrize("obstacle_type", ["table", "sofa", "shelf", "cabinet"])
def test_furniture_visual_types_keep_the_same_box_loss_calculation(obstacle_type):
    wall = box_obstacle("brick")
    furniture = wall.model_copy(update={"id": obstacle_type, "type": obstacle_type})
    point = np.array([[3.5, 0.5, 1.2]])
    wall_fields = evaluate_points(make_request(obstacles=[wall]), point)
    furniture_fields = evaluate_points(make_request(obstacles=[furniture]), point)
    assert furniture_fields["rssi_dbm"][0] == pytest.approx(wall_fields["rssi_dbm"][0])
    assert furniture_fields["obstacle_loss_db"][0] == pytest.approx(wall_fields["obstacle_loss_db"][0])


def test_obstacle_crossing_details_can_be_omitted_to_bound_response_size():
    request = make_request(include_obstacle_paths=False, obstacles=[box_obstacle("brick")])
    _, response = simulate(request)
    fields = response["fields"]
    assert "obstacle_crossings" not in fields
    assert "obstacle_ids" in fields
    assert "obstacle_count_zyx" in fields


def test_frequency_and_receiver_height_change_received_power():
    request = make_request()
    point_low = np.array([[3.5, 3.5, 0.5]])
    point_high = np.array([[3.5, 3.5, 2.0]])
    rssi_low = evaluate_points(request, point_low)["rssi_dbm"][0]
    rssi_high = evaluate_points(request, point_high)["rssi_dbm"][0]
    rssi_5 = evaluate_points(request, point_low, Frequency.GHZ_5)["rssi_dbm"][0]
    assert rssi_low != pytest.approx(rssi_high)
    assert rssi_5 < rssi_low


def test_transmit_power_changes_rssi_by_same_amount():
    point = np.array([[3.5, 3.5, 1.2]])
    first = evaluate_points(make_request(), point)["rssi_dbm"][0]
    second = evaluate_points(make_request(transmit_power_dbm=26), point)["rssi_dbm"][0]
    assert second - first == pytest.approx(6)


def test_snr_uses_bandwidth_noise_figure_and_linear_interference_sum():
    request = make_request(bandwidth_hz=20_000_000, noise_figure_db=7)
    point = np.array([[3.5, 3.5, 1.2]])
    fields = evaluate_points(request, point)
    assert fields["snr_db"][0] == pytest.approx(
        fields["rssi_dbm"][0] - thermal_noise_dbm(20_000_000, 7)
    )
    assert add_powers_dbm(-100, -100) == pytest.approx(-96.9897000434)
    with_interference = make_request(
        bandwidth_hz=20_000_000, noise_figure_db=7, interference_dbm=-90
    )
    assert evaluate_points(with_interference, point)["snr_db"][0] < fields["snr_db"][0]


def test_grid_has_z_y_x_axis_order_and_respects_voxel_limit():
    axes = grid_axes((4, 4, 2.4), 1)
    assert [len(axes[key]) for key in ("z", "y", "x")] == [2, 4, 4]
    result, response = simulate(make_request())
    assert result.shape == (2, 4, 4)
    assert response["grid"]["shape_zyx"] == [2, 4, 4]
    assert len(response["fields"]["rssi_dbm_zyx"]) == 2
    with pytest.raises(SimulationError, match="máximo"):
        grid_axes((100, 100, 30), 0.1)


def test_obstacle_voxels_are_invalid_and_solid_blocks_are_not_router_locations():
    obstacle = Obstacle.model_validate(
        {
            "id": "pillar",
            "center": {"x": 2, "y": 2, "z": 1.2},
            "dimensions": {"x": 1.5, "y": 1.5, "z": 2.4},
            "material": "concrete",
        }
    )
    request = make_request(obstacles=[obstacle])
    result, _ = simulate(request)
    assert result.valid.sum() < result.valid.size
    with pytest.raises(SimulationError, match="dentro de un obstáculo"):
        evaluate_points(
            request.model_copy(update={"router": Point3D(x=2, y=2, z=1.2)}),
            np.array([[3.5, 3.5, 1.2]]),
        )


def test_optimization_returns_clear_position_and_coverage_metrics():
    request = make_request(obstacles=[box_obstacle("concrete")])
    result = optimize_router(request)
    pos = result["recommended_position_m"]
    assert not _segment_intersects_boxes(
        np.array([pos["x"], pos["y"], pos["z"]]),
        np.array([[pos["x"], pos["y"], pos["z"]]]),
        request.obstacles[0],
    )[0]
    assert "coverage_change_percentage_points" in result


def test_pydantic_rejects_out_of_room_router_and_unordered_thresholds():
    with pytest.raises(ValueError):
        make_request(router={"x": 5, "y": 0.5, "z": 1})
    with pytest.raises(ValueError):
        make_request(
            thresholds={
                "dead_below_dbm": -50,
                "weak_below_dbm": -67,
                "fair_below_dbm": -60,
                "good_below_dbm": -75,
            }
        )


def test_symbolic_fspl_reference_matches_logarithmic_formula():
    symbolic_value = float(
        sp.N(
            fspl_db_expression.subs(
                {distance_km: 0.001, frequency_mhz: 2400}
            )
        )
    )
    assert symbolic_value == pytest.approx(float(fspl_db(1, 2.4)))
