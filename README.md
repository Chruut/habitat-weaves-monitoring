# Habitat Weaves – Web visualization 

Architecture for Biodiversity Installation - Bauhaus Foundation, 100 year Anniversary

Interactive **3D site overview** and linked **environmental data views** for the Habitat Weaves x Bauhaus Dessau, architecture for biodiversity research colaboration. The Web Browser experience is a Three.js viewer with an OBJ/MTL building model, sensor **heatmap-style overlays**, and navigation to companion dashboards.
<img width="384" height="512" alt="IMG-20260303-WA0015" src="https://github.com/user-attachments/assets/8358dd45-7e1f-4887-9c3b-03153fee7966" />

## Features

- **3D scan** [(polycam)](www.poly.cam): Environmental topography and texturing using photogrammetry
- **3D pipeline** (Python): Mesh decimation and asset prep (`decimate_model.py`) toward web-friendly OBJ/MTL and `interaction_points.json`.
- **Site viewer**: Orbit controls, fog/sky, station markers and tooltips, iframed sub-pages in the sidebar workflow.
- **Sensory Data Dashboard**: Integration of ESP32-based mesh of environmental sensors, field cameras and soil contact microphone recordings
- **Heatmaps** (Python): Soil temperature and humidity sensors distributed over the "Sandarium" habitat create a comprehensive coverage of year-round environmental conditions

### 3D viewer 
<img width="860" height="632" alt="chrome_DfGgwzH916" src="https://github.com/user-attachments/assets/0da5d2b9-ace0-408d-8850-50d8e5789711" />

### Sandarium Heatmap
<img width="887" height="553" alt="firefox_Xblc9ucWvR" src="https://github.com/user-attachments/assets/30313041-055a-49ce-a63c-2f3d6fbeb363" />

### Sensory Data Dashboards


## Tech stack

| Area | Stack |
|------|--------|
| Viewer | HTML/CSS, [Three.js](https://threejs.org/) (ES modules), OBJLoader |
| Backing services | Python 3.11+, [uv](https://github.com/astral-sh/uv), Panel, NumPy/Pandas, PyVista/Trimesh (where used) |

## Prerequisites

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) for installing dependencies from `pyproject.toml`

## Quick start (site viewer)

From the repository root:

```bash
uv sync
uv run python serve_site.py
```

Open **[http://localhost:8080/site_viewer.html](http://localhost:8080/site_viewer.html)**.

The viewer loads **`media/3D-objects/decimated/interaction_points.json`**, which defines OBJ/MTL paths and interaction metadata. Ensure that JSON and the referenced assets exist (or update paths after running your decimation/export workflow).

## Full stack (optional)

`start_all.py` launches the static site plus Panel apps and targets a Chromium kiosk URL. It assumes Linux-style paths for `uv` and the browser; on Windows, start services individually, for example:

```bash
uv run python serve_site.py
# In other terminals, as needed:
uv run panel serve sandarium_dashboard.py --port 5006 --allow-websocket-origin "*"
uv run panel serve camera_player.py --port 5007 --static-dirs video_db=./video_db --allow-websocket-origin "*"
uv run panel serve audio_player.py --port 5008 --static-dirs audio_db=./data/audio media_audio=./media/audio --allow-websocket-origin "*"
```

Ports **5006–5008** match the links wired from `site_viewer.html` (adjust if you change ports).

## Repository layout (high level)

| Path | Role |
|------|------|
| `site_viewer.html` | Main 3D shell and navigation |
| `pages/` | HTML fragments / embed pages linked from the sidebar |
| `media/3D-objects/` | OBJ, MTL, textures, decimated exports, `interaction_points.json` |
| `sandarium_dashboard.py`, `camera_player.py`, `audio_player.py` | Panel applications |
| `serve_site.py` | Local HTTP server for static files (port 8080) |
| `decimate_model.py` | Mesh processing helpers (logs under `logs/`) |
| `data/`, `video_db/`, `texts/` | Data and media inputs for dashboards and ingest scripts |

## Project metadata

- **Package name:** `web-viz` (see `pyproject.toml`)
- **Description:** HabitatWeaves data visualization – 3D heatmap dashboard and interactive site viewer

## License

Specify your license here if the project is public (this repository does not ship a default `LICENSE` file in this snapshot).
