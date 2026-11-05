# WiFiMap 3D simulation engine

The marketing landing page is served at `http://127.0.0.1:8000/`; open the functional simulator at `http://127.0.0.1:8000/simulador.html`. The API docs are available at `/docs`.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload
```

The engine works without MySQL. For optional XAMPP/MySQL persistence, create the database, run `backend/schema.sql`, and set the variables in `backend/.env.example` in the process environment before starting the API. Only simulation inputs and obstacle geometry are stored; voxel arrays remain bounded in memory.

## Publish the site and API

The GitHub Actions workflow in `.github/workflows/pages.yml` publishes `index.html` and `simulador.html` to GitHub Pages when changes are pushed to `main`. The backend is a separate Render web service defined by `render.yaml`; GitHub Pages cannot execute Python itself.

To finish the online setup:

1. In Render, create a new **Blueprint** from this GitHub repository and approve the `wifi-simulator-api` web service described by `render.yaml`.
2. Wait until Render reports the service healthy at `/api/health`, then copy its public HTTPS URL (for example, `https://wifi-simulator-api.onrender.com`).
3. In the GitHub repository, open **Settings → Secrets and variables → Actions → Variables**, add the repository variable `WIFIMAP_API_URL` with that HTTPS URL, and rerun **Deploy WiFiMap pages** from the Actions tab. The Pages workflow injects this URL into the generated `api-config.js`; do not include a trailing slash.
4. In **Settings → Pages**, select **GitHub Actions** as the build and deployment source if it is not already selected.

The `api-config.js` committed to the repository intentionally contains no service URL; the Pages deployment workflow generates the configured version. Locally, an empty API URL uses the same-origin FastAPI server.

## Configure a simulation in the page

From the landing page, select **Abrir el simulador** to open `simulador.html`. Set room dimensions, one to eight access-point radio positions, transmit powers, bands and channels, voxel size, and up to 20 obstacles with material and 3D dimensions. The initial values recreate the sample room. Submit the form to calculate the selected band; use the 2.4/5 GHz controls in the 3D view to recalculate the same setup at the other frequency. The result card reports average RSSI/SINR, useful coverage, voxel count, and the share of valid voxels served by each active radio.

Use **Optimizar canales** with at least two radios on the selected band to request a heuristic channel plan. The recommendation compares the current and proposed average/10th-percentile SINR; **Aplicar canales sugeridos** explicitly copies the proposed channels into the form and recalculates the map. The optimizer evaluates up to 1,000 voxels and performs up to three coordinate-descent passes over supported channels, so the result is a local recommendation, not a guaranteed global optimum.

After a successful calculation, enter a name and save the scenario to this browser. Up to 10 scenarios store the submitted configuration and summary metrics (not the 3D voxel arrays). Select two saved scenarios to compare RSSI, useful coverage, SINR, grid size, and AP channels; use **Cargar** to restore a saved setup and recalculate it before editing or saving a new result. Saved scenarios remain in the browser's local storage and are not synchronized to the API or another device.

The room and access-point coordinates use metres: X/Y are on the floor plane and Z is height. Obstacles are axis-aligned boxes. Keep every access point and obstacle inside the room; the interface checks the 20,000-voxel engine limit before sending a request. The 3D view draws room dimensions, the active-band access points, material-colored obstacle boxes, and three slices of the selected RSSI or SINR map.

## 3D materials and lighting

The browser scene builds tileable PBR-style color, roughness, and normal maps procedurally for the floor, room walls, and selected obstacle materials. Obstacles can be rendered as boxes, columns, tables, sofas, shelves, or cabinets. Furniture meshes remain inside the configured axis-aligned box, which is outlined in the scene to show the exact volume used by the calculation. Material appearance and furniture detail are visual only: obstacle loss calculations still use the configured material attenuation values and box intersections in the Python engine. Decorative wall trim and router details are not additional simulation obstacles.

The renderer uses a neutral hemisphere/key/fill light setup with soft directional shadows. RSSI heatmap slices use an unlit material with tone mapping disabled so lighting and surface reflections do not change the data-color scale. No external texture assets are downloaded; texture maps are generated in the browser and disposed when the scene is rebuilt or closed.

The simulator's API base URL defaults to the same origin. To connect a separately hosted API, set the existing `wifi-simulator-api` meta tag in `simulador.html` to its HTTPS origin and configure that page origin in `CORS_ORIGINS`. GitHub Pages serves the landing and simulator as static pages; deploying the Python API requires a separate Python-capable host.

## Coordinate and propagation model

Coordinates are metres with **X and Y on the floor plane and Z as height**. The response matrices always use `[z, y, x]`. Three.js uses Y as its vertical axis, so the existing scene adapter maps API coordinates `(x, y, z)` to Three.js `(x, z, y)`.

Received power is `Ptx + Gtx - Ltx - FSPL - obstacle losses - additional losses + Grx`. FSPL uses distance in kilometres and frequency in MHz. Distances below 0.1 m are clamped to 0.1 m to avoid a singularity. Thermal noise is `-174 + 10 log10(B) + NF`; configured interference is added in linear power units.

Obstacle geometry is an axis-aligned 3D box. Each distinct box contributes its configured material loss once when a direct source-to-voxel segment intersects it. The editor supports openings in elongated walls: a door is a clear passage and its voxels are not treated as solid; a window remains solid and replaces the wall loss with the estimated `glass` material loss. The opening width, horizontal offset, and height must fit inside the wall box. For multiple radios on the selected band, each voxel is associated with the radio having the highest estimated SINR. Interference is estimated by summing received powers in linear units; 2.4 GHz channels use a simplified overlap weight that decreases with channel separation, while 5 GHz interference is included only for equal channel numbers. The channel optimizer maximizes a weighted score of SINR's 10th percentile and mean on a reduced grid using coordinate descent. This is a channel-planning approximation, not a standards-accurate adjacent-channel or airtime model. Default material losses are **orientative estimates**, not experimental measurements; custom per-frequency losses may be supplied. Reflection, diffraction, furniture scattering, multipath, client roaming, and real-device channel behavior are not modeled.

Voxels inside a solid obstacle are marked invalid. Grid size is capped at 20,000 voxels per request. RSSI classes default to excellent (`>= -50 dBm`), good (`>= -60`), fair (`>= -67`), weak (`>= -75`) and dead (`< -75`); thresholds are configurable.

## API

- `GET /api/health`
- `GET /api/materiales`
- `POST /api/simulacion/calcular`
- `POST /api/simulacion/comparar`
- `POST /api/simulacion/optimizar`
- `POST /api/simulacion/optimizar-canales`
- `GET /api/simulacion/cortes?simulation_id=...&axis=z&coordinate_m=1.2`

Request JSON is validated by Pydantic. The calculate and compare responses contain numeric grids (unless `include_grid` is false) plus statistics and model limitations. A cut is available from the bounded in-memory cache; restart or eviction requires recalculation. The optimizer uses SciPy differential evolution over a reduced receiver grid and avoids candidates inside supplied solid obstacles.

Material attenuation defaults must be replaced with measurements for a calibrated deployment. This direct-path model is an educational estimate, not a standards-based indoor survey or a full electromagnetic solver.
