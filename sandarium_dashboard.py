"""Sandarium 3D Heatmap Dashboard — Proof of Concept.

Loads a 3D mesh of the sandarium, reads temperature data from 4 field probes,
interpolates (IDW) across the mesh surface, and renders as a heatmap with a
timeline slider.  Designed for Panel served locally on a Raspberry Pi 5 with
a 1920x1080 HDMI monitor.
"""

import io
import os
from pathlib import Path

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    print("Checkpoint: no DISPLAY — PYVISTA_OFF_SCREEN enabled")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import panel as pn
import pyvista as pv
import vtk
from scipy.spatial.distance import cdist


class NoZoomTrackballStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball rotate/pan; mouse wheel does not zoom."""

    def OnMouseWheelForward(self):
        pass

    def OnMouseWheelBackward(self):
        pass

pn.extension("vtk", sizing_mode="stretch_width")

_BASE = Path(__file__).resolve().parent
print(f"Checkpoint: base directory = {_BASE}")

# ---------------------------------------------------------------------------
# 1. Load mesh
# ---------------------------------------------------------------------------
STL_PATH = _BASE / "media" / "3D-objects" / "sandarium - II draped mesh-1.stl"
print(f"Checkpoint: loading mesh from {STL_PATH}")
mesh = pv.read(str(STL_PATH))
mesh.points -= mesh.center
mesh = mesh.clean()
print(f"Checkpoint: mesh after clean — {mesh.n_points} vertices, {mesh.n_cells} faces")

TARGET_FACES = 50_000
if mesh.n_cells > TARGET_FACES:
    mesh = mesh.decimate(1.0 - TARGET_FACES / mesh.n_cells)
    print(f"Checkpoint: decimated to {mesh.n_points} vertices, {mesh.n_cells} faces")

# ---------------------------------------------------------------------------
# 2. Probe positions — spread left to right, staggered vertically
# ---------------------------------------------------------------------------
PROBE_VERTEX_INDICES = [1163, 10994, 14332, 20982]
PROBE_LABELS = ["Lehm", "Sandiger Lehm", "Lehmiger Sand", "Sand"]
PROBE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
probe_positions = mesh.points[PROBE_VERTEX_INDICES]
print(f"Checkpoint: probe positions (centered coords):")
for i, (lbl, pos) in enumerate(zip(PROBE_LABELS, probe_positions)):
    print(f"  {lbl}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

# ---------------------------------------------------------------------------
# 3. Load synthetic CSV data
# ---------------------------------------------------------------------------
DATA_DIR = _BASE / "data" / "synthetic"
probe_temps = []
timestamps = None

for i in range(4):
    csv_path = DATA_DIR / f"probe_{i + 1}.csv"
    df = pd.read_csv(csv_path, skiprows=3, header=0)
    col_names = ["TIME", "TEMP", "HUM", "DALLAS_TEMP", "BATT", "LIGHT",
                 "NOISE_A", "PRESS", "CCS811_VOCS", "CCS811_ECO2",
                 "CHRP_MOIS_RAW", "CHRP_MOIS", "CHRP_TEMP", "CHRP_LIGHT"]
    df.columns = col_names
    temps = pd.to_numeric(df["DALLAS_TEMP"], errors="coerce").values
    probe_temps.append(temps)
    if timestamps is None:
        timestamps = pd.to_datetime(df["TIME"], errors="coerce")

probe_temps = np.array(probe_temps)  # shape (4, 288)
n_timesteps = probe_temps.shape[1]
print(f"Checkpoint: loaded {n_timesteps} timesteps from 4 probes")

# ---------------------------------------------------------------------------
# 4. IDW precomputation
# ---------------------------------------------------------------------------
print("Checkpoint: precomputing IDW interpolation for all timesteps...")
distances = cdist(mesh.points, probe_positions)
distances = np.maximum(distances, 1e-10)
IDW_POWER = 2
weights = 1.0 / distances ** IDW_POWER
weight_sums = weights.sum(axis=1, keepdims=True)

all_scalars = np.zeros((n_timesteps, mesh.n_points), dtype=np.float32)
for t in range(n_timesteps):
    vals = probe_temps[:, t]
    all_scalars[t] = (weights * vals[np.newaxis, :]).sum(axis=1) / weight_sums.ravel()

temp_min = float(np.nanmin(all_scalars))
temp_max = float(np.nanmax(all_scalars))
print(f"Checkpoint: IDW done — temp range [{temp_min:.1f}, {temp_max:.1f}] °C, "
      f"array shape {all_scalars.shape}, size {all_scalars.nbytes / 1e6:.1f} MB")

# ---------------------------------------------------------------------------
# 5. Initial mesh scalars — start at 02:00
# ---------------------------------------------------------------------------
INITIAL_FRAME = 24
mesh["Temperature"] = all_scalars[INITIAL_FRAME]
mesh.set_active_scalars("Temperature")

BG_COLOR = "#ffffff"

plotter = pv.Plotter()
plotter.set_background(BG_COLOR)
plotter.add_mesh(
    mesh,
    scalars="Temperature",
    cmap="coolwarm",
    clim=[temp_min, temp_max],
    show_scalar_bar=True,
    scalar_bar_args={"title": "Temperatur (°C)", "n_labels": 5},
)

sphere_radius = mesh.length / 80
for i, pos in enumerate(probe_positions):
    sphere = pv.Sphere(radius=sphere_radius, center=pos, theta_resolution=48, phi_resolution=48)
    plotter.add_mesh(sphere, color=PROBE_COLORS[i], label=PROBE_LABELS[i])

plotter.camera_position = "xy"
plotter.reset_camera()
cam = plotter.camera
focal = np.array(cam.focal_point)
pos = np.array(cam.position)
direction = pos - focal
dist = np.linalg.norm(direction)
cam.position = (focal[0], focal[1] - dist * 0.35, focal[2] + dist * 0.85)
cam.up = (0, 1, 0)
plotter.reset_camera_clipping_range()

_vtk_iren = plotter.ren_win.GetInteractor()
if _vtk_iren is not None:
    _vtk_iren.SetInteractorStyle(NoZoomTrackballStyle())
    print("Checkpoint: VTK interactor — zoom disabled")
else:
    print("Checkpoint: WARNING — no VTK interactor")

VIEWER_SIZE = 280
vtk_pane = pn.panel(
    plotter.ren_win,
    height=VIEWER_SIZE,
    width=VIEWER_SIZE,
    sizing_mode="fixed",
)
print(f"Checkpoint: VTK pane size {VIEWER_SIZE}x{VIEWER_SIZE}px")

# ---------------------------------------------------------------------------
# 6. Time-series plot (replaces slider)
# ---------------------------------------------------------------------------

def build_timeseries_plot(frame_idx: int) -> bytes:
    """Render the full-day temperature plot with a vertical time cursor."""
    fig, ax = plt.subplots(figsize=(7, 2.7), dpi=100)
    fig.patch.set_facecolor("#f5f7f5")
    ax.set_facecolor("#fafafa")

    ts_values = timestamps.values if timestamps is not None else np.arange(n_timesteps)

    for i in range(4):
        ax.plot(ts_values, probe_temps[i], color=PROBE_COLORS[i],
                linewidth=1.3, label=PROBE_LABELS[i], alpha=0.85)

    if timestamps is not None and frame_idx < len(timestamps):
        cursor_x = timestamps.iloc[frame_idx]
        ax.axvline(x=cursor_x, color="black", linewidth=1.8, linestyle="-", zorder=10)

    ax.set_ylabel("°C", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3)

    if timestamps is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        fig.autofmt_xdate(rotation=0, ha="center")

    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


plot_pane = pn.pane.PNG(build_timeseries_plot(INITIAL_FRAME), width=700, sizing_mode="scale_width")

# ---------------------------------------------------------------------------
# 7. Timeline controls
# ---------------------------------------------------------------------------
frame_slider = pn.widgets.IntSlider(
    name="Zeitschritt",
    start=0,
    end=n_timesteps - 1,
    step=1,
    value=INITIAL_FRAME,
    visible=False,
)

STEP_5MIN = 1
STEP_1H = 12

time_display = pn.pane.Markdown("", width=300, margin=(0, 0, 0, 0))

btn_back_1h = pn.widgets.Button(name="<<", button_type="default", width=55)
btn_back_5 = pn.widgets.Button(name="<", button_type="default", width=55)
btn_fwd_5 = pn.widgets.Button(name=">", button_type="default", width=55)
btn_fwd_1h = pn.widgets.Button(name=">>", button_type="default", width=55)
nav_hint = pn.pane.Markdown(
    "`<<` / `>>` = ±1 h · `<` / `>` = ±5 min",
    width=260,
    styles={"font-size": "11px", "color": "#888"},
)


def format_time(frame_idx):
    if timestamps is not None and frame_idx < len(timestamps):
        ts = timestamps.iloc[frame_idx]
        if pd.notna(ts):
            return f"### {ts.strftime('%Y-%m-%d  %H:%M')} UTC"
    return f"### Frame {frame_idx}"


def _clamp_frame(n: int) -> int:
    return max(0, min(n_timesteps - 1, n))


def update_frame(event):
    idx = event.new
    mesh["Temperature"] = all_scalars[idx]
    mesh.set_active_scalars("Temperature")
    vtk_pane.param.trigger("object")
    time_display.object = format_time(idx)
    plot_pane.object = build_timeseries_plot(idx)
    probe_info.object = probe_info_md(idx)


def _nav(delta: int):
    frame_slider.value = _clamp_frame(frame_slider.value + delta)


btn_back_1h.on_click(lambda _e: _nav(-STEP_1H))
btn_back_5.on_click(lambda _e: _nav(-STEP_5MIN))
btn_fwd_5.on_click(lambda _e: _nav(STEP_5MIN))
btn_fwd_1h.on_click(lambda _e: _nav(STEP_1H))

frame_slider.param.watch(update_frame, "value")

time_display.object = format_time(INITIAL_FRAME)

# ---------------------------------------------------------------------------
# 8. Probe info card (simplified)
# ---------------------------------------------------------------------------
def probe_info_md(frame_idx):
    scalars = all_scalars[frame_idx]
    s_min, s_max = float(np.nanmin(scalars)), float(np.nanmax(scalars))
    lines = [
        "### Sondenwerte\n",
        "| Sonde | Temp (°C) |",
        "|-------|-----------|",
    ]
    for i, lbl in enumerate(PROBE_LABELS):
        val = probe_temps[i, frame_idx]
        dot = f'<span style="color:{PROBE_COLORS[i]}">●</span>'
        lines.append(f"| {dot} {lbl} | {val:.2f} |")
    lines.extend([
        "",
        "### Heatmap-Legende",
        f"- **Minimalwert:** {s_min:.2f} °C",
        f"- **Maximalwert:** {s_max:.2f} °C",
    ])
    return "\n".join(lines)


probe_info = pn.pane.Markdown(probe_info_md(INITIAL_FRAME), width=180)

# ---------------------------------------------------------------------------
# 9. Layout
# ---------------------------------------------------------------------------
logo_path = _BASE / "media" / "HW_logo_signage_v1.png"

BACK_LINK = pn.pane.HTML(
    '<a href="http://localhost:8080/site_viewer.html" '
    'style="color:#88d8b0; text-decoration:none; font-size:13px; '
    'display:inline-flex; align-items:center; gap:4px;">'
    '&larr; Site Overview</a>',
    height=24,
)

sidebar_items = [BACK_LINK]
if logo_path.exists():
    sidebar_items.append(pn.pane.PNG(str(logo_path), sizing_mode="scale_both", max_height=180))
sidebar_items.extend([
    pn.pane.Markdown("## Sandarium\nTemperatur-Heatmap"),
    pn.layout.Divider(),
    probe_info,
])

template = pn.template.FastListTemplate(
    title="Sandarium Heatmap Dashboard",
    sidebar=sidebar_items,
    sidebar_width=220,
    main=[
        pn.Column(
            pn.Row(
                pn.Column(
                    pn.layout.VSpacer(),
                    time_display,
                    pn.Row(btn_back_1h, btn_back_5, btn_fwd_5, btn_fwd_1h, margin=(0, 0, 0, 0)),
                    nav_hint,
                    margin=(0, 0, 0, 0),
                    height=VIEWER_SIZE,
                ),
                pn.layout.HSpacer(),
                vtk_pane,
                margin=(0, 0, 0, 0),
            ),
            plot_pane,
            margin=(0, 0, 0, 0),
        ),
    ],
    accent_base_color="#88d8b0",
    header_background="forestgreen",
)

template.servable()
print("Checkpoint: dashboard template defined and marked servable")
