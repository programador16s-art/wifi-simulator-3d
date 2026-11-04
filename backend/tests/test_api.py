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


def test_compare_uses_both_bands_with_same_geometry():
    response = client.post("/api/simulacion/comparar", json=SMALL_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["same_geometry_and_transmitter"] is True
    assert set(body["comparison"]) == {"2.4", "5"}


def test_optimizer_endpoint_returns_xyz_recommendation():
    response = client.post("/api/simulacion/optimizar", json=SMALL_REQUEST)
    assert response.status_code == 200
    result = response.json()
    assert set(result["recommended_position_m"]) == {"x", "y", "z"}
    assert result["evaluation_voxels"] > 0


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
