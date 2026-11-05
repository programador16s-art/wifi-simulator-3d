from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
SMALL_REQUEST = {
    "dimensions": {"x": 4, "y": 4, "z": 2},
    "router": {"x": 0.5, "y": 0.5, "z": 1},
    "voxel_size_m": 1,
}


def test_health_and_material_catalog():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    materials = client.get("/api/materiales")
    assert materials.status_code == 200
    assert materials.json()["materials"][0]["source"].startswith("Estimación")


def test_calculate_and_request_cached_cut():
    result = client.post("/api/simulacion/calcular", json=SMALL_REQUEST)
    assert result.status_code == 200
    data = result.json()
    assert data["grid"]["shape_zyx"] == [2, 4, 4]
    assert len(data["fields"]["rssi_dbm_zyx"]) == 2
    assert data["fields"]["obstacle_ids"] == []
    cut = client.get(
        "/api/simulacion/cortes",
        params={
            "simulation_id": data["simulation_id"],
            "axis": "z",
            "coordinate_m": 0.5,
        },
    )
    assert cut.status_code == 200
    assert len(cut.json()["rssi_dbm"]) == 4


def test_calculate_accepts_multiple_access_points_and_returns_service_map():
    request = {
        "dimensions": {"x": 4, "y": 1, "z": 1},
        "router": {"x": 0.5, "y": 0.5, "z": 0.5},
        "voxel_size_m": 1,
        "access_points": [
            {"id": "ap-left", "position": {"x": 0.5, "y": 0.5, "z": 0.5}},
            {"id": "ap-right", "position": {"x": 3.5, "y": 0.5, "z": 0.5}},
        ],
    }
    response = client.post("/api/simulacion/calcular", json=request)
    assert response.status_code == 200
    data = response.json()
    assert [point["id"] for point in data["access_points"]] == ["ap-left", "ap-right"]
    assert data["fields"]["serving_ap_index_zyx"][0][0] == [0, 0, 1, 1]


def test_compare_uses_both_bands_with_same_geometry():
    response = client.post("/api/simulacion/comparar", json=SMALL_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["same_geometry_and_transmitter"] is True
    assert set(body["comparison"]) == {"2.4", "5"}


def test_compare_reports_only_bands_with_configured_radios():
    request = {
        **SMALL_REQUEST,
        "access_points": [
            {
                "id": "ap-24",
                "position": {"x": 0.5, "y": 0.5, "z": 1},
                "frequency": "2.4",
                "channel": 1,
            }
        ],
    }
    response = client.post("/api/simulacion/comparar", json=request)
    assert response.status_code == 200
    body = response.json()
    assert body["available_frequencies_ghz"] == ["2.4"]
    assert set(body["comparison"]) == {"2.4"}


def test_optimizer_endpoint_returns_xyz_recommendation():
    response = client.post("/api/simulacion/optimizar", json=SMALL_REQUEST)
    assert response.status_code == 200
    result = response.json()
    assert set(result["recommended_position_m"]) == {"x", "y", "z"}
    assert result["evaluation_voxels"] > 0


def test_channel_optimizer_endpoint_returns_before_and_after_plan():
    request = {
        "dimensions": {"x": 4, "y": 1, "z": 1},
        "router": {"x": 0.5, "y": 0.5, "z": 0.5},
        "voxel_size_m": 0.5,
        "access_points": [
            {
                "id": "ap-left",
                "position": {"x": 0.5, "y": 0.5, "z": 0.5},
                "frequency": "2.4",
                "channel": 1,
            },
            {
                "id": "ap-right",
                "position": {"x": 3.5, "y": 0.5, "z": 0.5},
                "frequency": "2.4",
                "channel": 1,
            },
        ],
    }
    response = client.post("/api/simulacion/optimizar-canales", json=request)
    assert response.status_code == 200
    data = response.json()
    assert data["after"]["average_sinr_db"] > data["before"]["average_sinr_db"]
    assert len({item["channel"] for item in data["recommended_channels"]}) == 2


def test_invalid_request_and_unknown_cut_report_errors():
    invalid = {**SMALL_REQUEST, "router": {"x": 8, "y": 0.5, "z": 1}}
    assert client.post("/api/simulacion/calcular", json=invalid).status_code == 422
    assert client.get(
        "/api/simulacion/cortes",
        params={"simulation_id": "missing", "axis": "z", "coordinate_m": 1},
    ).status_code == 404


def test_landing_and_simulator_pages_are_separate():
    response = client.get("/")
    assert response.status_code == 200
    assert "Abrir el simulador" in response.text
    assert 'href="simulador.html"' in response.text
    assert 'id="simulation-form"' not in response.text

    simulator = client.get("/simulador.html")
    assert simulator.status_code == 200
    assert 'id="simulation-form"' in simulator.text
    assert client.get("/simulador").status_code == 200
    config = client.get("/api-config.js")
    assert config.status_code == 200
    assert 'window.WIFI_SIMULATOR_API = "";' in config.text
