"""Decimate the full-site OBJ model for web viewing on Raspberry Pi.

Source mesh (OBJ + MTL in the same folder): by default
``media/3D-objects/new-3D-models/full site_ ove data vis _ with sandarium 3D scan.obj``.
Outputs ``site_decimated.obj``, ``site_decimated.mtl``, textures, and
``interaction_points.json`` under ``media/3D-objects/decimated/`` for ``site_viewer.html``.
Diffuse maps listed in the MTL are copied from ``media/3D-objects`` and/or the OBJ folder.

Uses trimesh + fast-simplification for decimation while preserving
material definitions. Outputs OBJ+MTL so material colors (and textures,
when the referenced image files are present) are retained on the
decimated geometry.

The raw Rhino OBJ often ends with a NURBS appendix (``cstype bspline``, ``curv``,
…). **three.js OBJLoader cannot parse that** and will warn and produce NaN
geometry. For full-resolution web viewing we export ``site_threejs_full.obj`` —
triangle mesh only from trimesh, same triangles as this tool loads from the
Rhino file.

Recommended starting points for Raspberry Pi 5 with *full site* (~491 MB):
  95% → ~135K faces (good balance for the large model)
  97% → ~81K faces  (faster, slightly coarser)
  98% → ~54K faces  (very fast, noticeably simplified; safer for Firefox)

For **Firefox** and other browsers, prefer ``site_decimated.obj`` (this file).
``site_threejs_full.obj`` is often hundreds of MB as text — far beyond what is
reliably loadable in Firefox (no hard cap; practical issues typically appear
above roughly tens of MB for single-threaded OBJ text decode + GPU upload).
"""

# =========================================================================
# CONFIGURATION — change this value and re-run to test different levels
# =========================================================================
# Decimated preview mesh (``site_decimated.obj``) for Pi / Firefox-friendly size.
REDUCTION_PERCENT = 98
# True: viewer loads full triangle count as ``site_threejs_full.obj`` in new-3D-models (not decimated).
VIEWER_USE_FULL_RES_SOURCE_MESH = True
# =========================================================================

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
# Primary export used for the home overview (`site_viewer.html`).
SITE_MODEL_DIR = BASE / "media" / "3D-objects" / "new-3D-models"
# OBJ/MTL live under ``new-3D-models``; referenced JPGs are often in ``media/3D-objects``.
MODEL_OBJECTS_ROOT = SITE_MODEL_DIR.parent
OBJ_PATH = SITE_MODEL_DIR / "full site_ ove data vis _ with sandarium 3D scan.obj"
MTL_SRC = OBJ_PATH.with_suffix(".mtl")
OUT_DIR = BASE / "media" / "3D-objects" / "decimated"
PAGES_DIR = BASE / "pages"
LOG_DIR = BASE / "logs"

OUT_OBJ_NAME = "site_decimated.obj"
OUT_MTL_NAME = "site_decimated.mtl"
THREEJS_FULL_OBJ_NAME = "site_threejs_full.obj"
THREEJS_FULL_MTL_NAME = "site_threejs_full.mtl"
# Three.js-safe full mesh (same triangles as scan; no Rhino NURBS tail) lives beside the source OBJ.
THREEJS_FULL_DIR = SITE_MODEL_DIR
VIEWER_THREEJS_MESH_BASE = "media/3D-objects/new-3D-models/"
VIEWER_THREEJS_TEX_PATH = "media/3D-objects/new-3D-models/"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "decimate_model.log"),
    ],
)
log = logging.getLogger(__name__)

STATIONS = [
    {"name": "Sandarium",                "color": "#e74c3c", "category": "Environmental Data",  "page": "sandarium.html"},
    {"name": "Camera \u2013 Sandarium",        "color": "#3498db", "category": "Camera",              "page": "camera-sandarium.html"},
    {"name": "Camera \u2013 Schatten",          "color": "#3498db", "category": "Camera",              "page": "camera-schatten.html"},
    {"name": "Chromatography \u2013 Spot A",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-a.html"},
    {"name": "Chromatography \u2013 Spot B",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-b.html"},
    {"name": "Chromatography \u2013 Spot C",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-c.html"},
    {"name": "Chromatography \u2013 Spot D",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-d.html"},
    {"name": "Chromatography \u2013 Spot E",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-e.html"},
    {"name": "Chromatography \u2013 Spot F",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-f.html"},
    {"name": "Chromatography \u2013 Spot G",    "color": "#8B4513", "category": "Soil Chromatography", "page": "chromatography-g.html"},
    {"name": "Soil DJ",                    "color": "#e67e22", "category": "Audio",               "page": "audio-recording.html"},
]


def mtl_referenced_texture_basenames(mtl_path: Path) -> list[str]:
    """Basenames from ``map_*`` lines (e.g. ``map_Kd foo.jpg``) in an MTL file."""
    if not mtl_path.is_file():
        return []
    seen: dict[str, None] = {}
    for line in mtl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.startswith("map_"):
            continue
        parts = s.split(None, 1)
        if len(parts) >= 2:
            seen.setdefault(Path(parts[1].strip()).name, None)
    return list(seen.keys())


def copy_referenced_textures(mtl_src: Path, out_dir: Path, search_roots: list[Path]) -> int:
    """Copy each MTL-referenced image from the first matching ``search_roots`` path."""
    names = mtl_referenced_texture_basenames(mtl_src)
    if not names:
        return 0
    n = 0
    for name in names:
        src = None
        for root in search_roots:
            if not root.is_dir():
                continue
            cand = root / name
            if cand.is_file():
                src = cand
                break
        if src is not None:
            shutil.copy2(src, out_dir / name)
            log.info("  Copied texture: %s <- %s", name, src)
            n += 1
        else:
            log.warning(
                "  Referenced texture not found: %s (searched: %s)",
                name,
                ", ".join(str(r) for r in search_roots if r.is_dir()),
            )
    return n


def sample_spread_points(face_centers: np.ndarray, n: int, seed: int = 42) -> list[int]:
    """Furthest-point sampling for well-spread placement across the surface."""
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(face_centers)))]
    min_distances = np.full(len(face_centers), np.inf)

    for _ in range(n - 1):
        last = face_centers[selected[-1]]
        d = np.linalg.norm(face_centers - last, axis=1)
        min_distances = np.minimum(min_distances, d)
        selected.append(int(np.argmax(min_distances)))

    return selected


def scan_obj_materials(obj_path: Path) -> list[tuple[str, int]]:
    """Scan an OBJ file for unique materials and their triangulated face counts.

    Returns a list of (material_name, tri_face_count) in order of first
    appearance.  Quads and n-gons are counted as (n-2) triangles to match
    what trimesh produces after triangulation.
    """
    mat_faces: dict[str, int] = {}
    mat_order: list[str] = []
    current: str | None = None

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("usemtl "):
                name = line[7:].strip()
                if name not in mat_faces:
                    mat_faces[name] = 0
                    mat_order.append(name)
                current = name
            elif line.startswith("f ") and current is not None:
                n_verts = len(line.split()) - 1
                mat_faces[current] += max(n_verts - 2, 1)

    return [(name, mat_faces[name]) for name in mat_order]


def match_submeshes_to_materials(
    geometries: dict[str, trimesh.Trimesh],
    mat_info: list[tuple[str, int]],
) -> dict[str, str]:
    """Match trimesh sub-mesh names to original material names by face count."""
    geom_faces = {name: m.faces.shape[0] for name, m in geometries.items()}
    mat_faces = {name: count for name, count in mat_info}

    mapping: dict[str, str] = {}
    used_mats: set[str] = set()

    for gname, gcount in geom_faces.items():
        best_mat = None
        best_diff = float("inf")
        for mname, mcount in mat_faces.items():
            if mname in used_mats:
                continue
            diff = abs(gcount - mcount)
            if diff < best_diff:
                best_diff = diff
                best_mat = mname
        if best_mat is not None:
            mapping[gname] = best_mat
            used_mats.add(best_mat)
        else:
            mapping[gname] = "Default"

    return mapping


def recover_uvs(orig_verts: np.ndarray, orig_uvs: np.ndarray,
                dec_verts: np.ndarray) -> np.ndarray:
    """Map UVs from original mesh to decimated mesh via nearest-neighbor lookup."""
    tree = cKDTree(orig_verts)
    _, indices = tree.query(dec_verts)
    return orig_uvs[indices]


def get_mesh_uvs(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Extract per-vertex UV array from a trimesh, or None if absent."""
    vis = getattr(mesh, "visual", None)
    if vis is None:
        return None
    if hasattr(vis, "uv") and vis.uv is not None:
        uv = np.asarray(vis.uv)
        if uv.shape[0] == mesh.vertices.shape[0] and uv.shape[1] >= 2:
            return uv[:, :2]
    return None


def write_obj(
    meshes_dict: dict[str, trimesh.Trimesh],
    uv_data: dict[str, np.ndarray | None],
    mat_mapping: dict[str, str],
    obj_path: Path,
    mtl_name: str,
    *,
    header_comment: str | None = None,
) -> None:
    """Write sub-meshes as one OBJ referencing ``mtl_name`` (streaming I/O)."""
    if header_comment is None:
        header_comment = f"Decimated model ({REDUCTION_PERCENT}% reduction)"
    with Path(obj_path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {header_comment}\n")
        f.write(f"mtllib {mtl_name}\n\n")
        vertex_offset = 0
        uv_offset = 0
        vn_offset = 0
        for geom_name, mesh in meshes_dict.items():
            mat_name = mat_mapping.get(geom_name, geom_name)
            uvs = uv_data.get(geom_name)

            f.write(f"o {geom_name}\n")
            f.write(f"usemtl {mat_name}\n")

            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if uvs is not None:
                for uv in uvs:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            norms = np.asarray(mesh.vertex_normals, dtype=np.float64)
            for n in norms:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            if uvs is not None:
                for face in mesh.faces:
                    vi0 = face[0] + 1 + vertex_offset
                    vi1 = face[1] + 1 + vertex_offset
                    vi2 = face[2] + 1 + vertex_offset
                    ti0 = face[0] + 1 + uv_offset
                    ti1 = face[1] + 1 + uv_offset
                    ti2 = face[2] + 1 + uv_offset
                    ni0 = face[0] + 1 + vn_offset
                    ni1 = face[1] + 1 + vn_offset
                    ni2 = face[2] + 1 + vn_offset
                    f.write(
                        f"f {vi0}/{ti0}/{ni0} {vi1}/{ti1}/{ni1} {vi2}/{ti2}/{ni2}\n"
                    )
                uv_offset += len(uvs)
            else:
                for face in mesh.faces:
                    vi0 = face[0] + 1 + vertex_offset
                    vi1 = face[1] + 1 + vertex_offset
                    vi2 = face[2] + 1 + vertex_offset
                    ni0 = face[0] + 1 + vn_offset
                    ni1 = face[1] + 1 + vn_offset
                    ni2 = face[2] + 1 + vn_offset
                    f.write(f"f {vi0}//{ni0} {vi1}//{ni1} {vi2}//{ni2}\n")

            n_verts = len(mesh.vertices)
            vertex_offset += n_verts
            vn_offset += n_verts
            f.write("\n")


def load_centered_geometries() -> tuple[
    dict[str, trimesh.Trimesh],
    dict[str, str],
    np.ndarray,
    int,
    list[tuple[str, int]],
]:
    """Load Rhino OBJ with trimesh, match materials, center verts; return (geometries, …)."""
    log.info("Scanning OBJ for material definitions ...")
    print(f"[checkpoint] Scanning materials in {OBJ_PATH}")
    mat_info = scan_obj_materials(OBJ_PATH)
    for mname, mcount in mat_info:
        log.info("  material %-25s  %d tri-faces", mname, mcount)

    log.info("Loading OBJ geometry from %s", OBJ_PATH)
    print(f"[checkpoint] Starting OBJ load: {OBJ_PATH}")

    scene_or_mesh = trimesh.load(str(OBJ_PATH), process=False)

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        geometries: dict[str, trimesh.Trimesh] = {"mesh_0": scene_or_mesh}
    elif isinstance(scene_or_mesh, trimesh.Scene):
        geometries = dict(scene_or_mesh.geometry)
    else:
        log.error("Unexpected type from trimesh.load: %s", type(scene_or_mesh))
        sys.exit(1)

    total_faces_orig = sum(m.faces.shape[0] for m in geometries.values())
    total_verts_orig = sum(m.vertices.shape[0] for m in geometries.values())
    log.info(
        "Loaded %d sub-mesh(es): %s verts, %s faces",
        len(geometries),
        f"{total_verts_orig:,}",
        f"{total_faces_orig:,}",
    )
    print(
        f"[checkpoint] Loaded: {len(geometries)} sub-meshes, "
        f"{total_verts_orig:,} verts, {total_faces_orig:,} faces"
    )

    mat_mapping = match_submeshes_to_materials(geometries, mat_info)
    for gname, mname in mat_mapping.items():
        mesh = geometries[gname]
        log.info(
            "  sub-mesh %-25s  faces=%-8d  -> %s",
            gname,
            mesh.faces.shape[0],
            mname,
        )

    all_verts = np.vstack([m.vertices for m in geometries.values()])
    center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2.0
    del all_verts
    log.info("Centering offset: (%.1f, %.1f, %.1f)", *center)

    for mesh in geometries.values():
        mesh.vertices -= center

    return geometries, mat_mapping, center, total_faces_orig, mat_info


def generate_stub_page(station: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{station["name"]} \u2013 HabitatWeaves</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7f5; color: #333; }}
  .container {{ max-width: 800px; margin: 0 auto; padding: 40px 20px; }}
  .back-link {{
    display: inline-flex; align-items: center; gap: 6px;
    margin-bottom: 24px; color: #2c7a4f; text-decoration: none;
    font-weight: 500;
  }}
  .back-link:hover {{ text-decoration: underline; }}
  h1 {{ color: #1a3a2a; margin-bottom: 12px; }}
  .badge {{
    display: inline-block; padding: 4px 14px; border-radius: 12px;
    font-size: 13px; color: white; margin-bottom: 24px;
  }}
  .card {{
    background: white; border-radius: 14px; padding: 32px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }}
  .card p {{ line-height: 1.7; color: #555; }}
  .thumbnail-area {{
    width: 100%; height: 200px; background: #eef2ee; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: #999; margin-top: 20px; font-size: 14px;
  }}
</style>
</head>
<body>
<div class="container">
  <a class="back-link" href="../site_viewer.html">&larr; Back to Site Overview</a>
  <h1>{station["name"]}</h1>
  <span class="badge" style="background:{station["color"]}">{station["category"]}</span>
  <div class="card">
    <p>Detailed information about this station will be added here.</p>
    <div class="thumbnail-area">Thumbnail image placeholder</div>
  </div>
</div>
</body>
</html>"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    geometries, mat_mapping, center, total_faces_orig, _mat_info = (
        load_centered_geometries()
    )
    texture_roots = [MODEL_OBJECTS_ROOT, OBJ_PATH.parent]

    if VIEWER_USE_FULL_RES_SOURCE_MESH:
        THREEJS_FULL_DIR.mkdir(parents=True, exist_ok=True)
        uv_full = {name: get_mesh_uvs(mesh) for name, mesh in geometries.items()}
        threejs_obj = THREEJS_FULL_DIR / THREEJS_FULL_OBJ_NAME
        log.info("Writing Three.js-safe full mesh to %s", threejs_obj)
        print(
            "[checkpoint] Exporting Three.js-safe full mesh (trimesh only; "
            "Rhino NURBS appendix omitted) — may take several minutes"
        )
        write_obj(
            geometries,
            uv_full,
            mat_mapping,
            threejs_obj,
            THREEJS_FULL_MTL_NAME,
            header_comment=(
                "Full-resolution mesh via trimesh (Three.js-compatible; "
                "Rhino NURBS appendix stripped)"
            ),
        )
        threejs_mtl = THREEJS_FULL_DIR / THREEJS_FULL_MTL_NAME
        if MTL_SRC.exists():
            shutil.copy2(MTL_SRC, threejs_mtl)
            log.info("Copied MTL to %s", threejs_mtl)
        print(f"[checkpoint] Copying textures for full-mesh MTL -> {THREEJS_FULL_DIR}")
        copy_referenced_textures(MTL_SRC, THREEJS_FULL_DIR, texture_roots)

    # ── Decimate each sub-mesh ────────────────────────────────────────
    target_ratio = 1.0 - REDUCTION_PERCENT / 100.0
    decimated: dict[str, trimesh.Trimesh] = {}

    print(f"[checkpoint] Starting {REDUCTION_PERCENT}% decimation "
          f"(target ratio {target_ratio:.2f}) ...")

    uv_data: dict[str, np.ndarray | None] = {}

    for name, mesh in geometries.items():
        orig_faces = mesh.faces.shape[0]
        target_count = max(int(orig_faces * target_ratio), 4)
        orig_uvs = get_mesh_uvs(mesh)

        log.info("  Decimating '%s' (%s): %d -> target %d faces  [UVs: %s] ...",
                 name, mat_mapping.get(name, "?"), orig_faces, target_count,
                 "yes" if orig_uvs is not None else "no")

        orig_verts = np.asarray(mesh.vertices, dtype=np.float64)
        pts_out, tri_out = fast_simplification.simplify(
            points=np.asarray(mesh.vertices, dtype=np.float32),
            triangles=np.asarray(mesh.faces, dtype=np.int32),
            target_count=target_count,
        )
        simple = trimesh.Trimesh(vertices=pts_out, faces=tri_out, process=False)
        decimated[name] = simple

        if orig_uvs is not None:
            recovered = recover_uvs(orig_verts, orig_uvs,
                                    np.asarray(pts_out, dtype=np.float64))
            uv_data[name] = recovered
            log.info("    -> %d faces, UVs recovered via nearest-neighbor",
                     simple.faces.shape[0])
        else:
            uv_data[name] = None
            log.info("    -> %d faces (%.1f%% reduction)",
                     simple.faces.shape[0],
                     (1 - simple.faces.shape[0] / max(orig_faces, 1)) * 100)

    total_dec = sum(m.faces.shape[0] for m in decimated.values())
    log.info("Total decimated: %s faces (from %s)",
             f"{total_dec:,}", f"{total_faces_orig:,}")
    print(f"[checkpoint] Decimation complete: {total_dec:,} faces")

    # ── Write OBJ ─────────────────────────────────────────────────────
    obj_path = OUT_DIR / OUT_OBJ_NAME
    write_obj(decimated, uv_data, mat_mapping, obj_path, OUT_MTL_NAME)
    obj_mb = obj_path.stat().st_size / 1e6
    log.info("Saved OBJ to %s (%.1f MB)", obj_path, obj_mb)

    # ── Copy MTL ──────────────────────────────────────────────────────
    mtl_path = OUT_DIR / OUT_MTL_NAME
    if MTL_SRC.exists():
        shutil.copy2(MTL_SRC, mtl_path)
        log.info("Copied MTL to %s", mtl_path)
    else:
        log.warning("MTL source not found: %s", MTL_SRC)

    # ── Copy texture files (MTL ``map_*`` basenames; search parent ``media/3D-objects`` first)
    print(f"[checkpoint] Resolving textures from MTL refs in {MTL_SRC.name}; roots={texture_roots}")
    tex_count = copy_referenced_textures(MTL_SRC, OUT_DIR, texture_roots)
    if tex_count == 0 and mtl_referenced_texture_basenames(MTL_SRC):
        log.warning(
            "MTL lists texture maps but none were copied — check files exist under %s",
            texture_roots,
        )
    elif tex_count == 0:
        log.info("No map_* texture references in MTL — diffuse colors only.")
    print(f"[checkpoint] Copied {tex_count} referenced texture file(s) -> {OUT_DIR}")

    # ── Remove old STL if present ─────────────────────────────────────
    old_stl = OUT_DIR / "site_decimated.stl"
    if old_stl.exists():
        old_stl.unlink()
        log.info("Removed old STL: %s", old_stl.name)

    # ── Interaction points ────────────────────────────────────────────
    log.info("Sampling %d interaction points ...", len(STATIONS))
    combined = trimesh.util.concatenate(list(decimated.values()))
    face_centers = combined.triangles_center

    indices = sample_spread_points(face_centers, len(STATIONS))

    stations_out = []
    for i, station in enumerate(STATIONS):
        pos = face_centers[indices[i]]
        entry = {**station, "position": [float(pos[0]), float(pos[1]), float(pos[2])]}
        stations_out.append(entry)
        log.info("  %s: (%.1f, %.1f, %.1f)", station["name"], *pos)

    if VIEWER_USE_FULL_RES_SOURCE_MESH:
        viewer_model = {
            "file": THREEJS_FULL_OBJ_NAME,
            "mtl_file": THREEJS_FULL_MTL_NAME,
            "mesh_base": VIEWER_THREEJS_MESH_BASE,
            "texture_path": VIEWER_THREEJS_TEX_PATH,
            "needs_center_offset": False,
        }
        log.info(
            "interaction_points.json uses %s (Three.js-safe full mesh). "
            "Set VIEWER_USE_FULL_RES_SOURCE_MESH=False for site_decimated.obj",
            THREEJS_FULL_OBJ_NAME,
        )
        print(
            f"[checkpoint] Viewer will load {THREEJS_FULL_OBJ_NAME};",
            "set VIEWER_USE_FULL_RES_SOURCE_MESH=False for decimated OBJ",
        )
    else:
        viewer_model = {
            "file": OUT_OBJ_NAME,
            "mtl_file": OUT_MTL_NAME,
            "mesh_base": "media/3D-objects/decimated/",
            "texture_path": "media/3D-objects/decimated/",
            "needs_center_offset": False,
        }

    output_data = {
        "model": {
            **viewer_model,
            "format": "obj",
            "original_faces": total_faces_orig,
            "decimated_faces": total_dec,
            "reduction_percent": REDUCTION_PERCENT,
            "center_offset": [float(c) for c in center],
        },
        "stations": stations_out,
    }

    json_path = OUT_DIR / "interaction_points.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info("Saved interaction points to %s", json_path)
    print(f"[checkpoint] Wrote {json_path}")

    # ── Stub pages ────────────────────────────────────────────────────
    log.info("Generating %d stub subpages in %s ...", len(STATIONS), PAGES_DIR)
    for station in STATIONS:
        page_path = PAGES_DIR / station["page"]
        page_path.write_text(generate_stub_page(station), encoding="utf-8")
        log.info("  Created %s", page_path)

    log.info("All done!")
    log.info("  1. Run:  python serve_site.py")
    log.info("  2. Open: http://localhost:8080/site_viewer.html")
    print("[checkpoint] Pipeline finished successfully")


def export_threejs_full_only() -> None:
    """Write ``site_threejs_full.obj`` under new-3D-models and point ``interaction_points.json`` at it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THREEJS_FULL_DIR.mkdir(parents=True, exist_ok=True)
    geometries, mat_mapping, center, total_faces_orig, _mat_info = (
        load_centered_geometries()
    )
    uv_full = {name: get_mesh_uvs(mesh) for name, mesh in geometries.items()}
    threejs_obj = THREEJS_FULL_DIR / THREEJS_FULL_OBJ_NAME
    log.info("Writing Three.js-safe full mesh to %s", threejs_obj)
    print(
        "[checkpoint] export-threejs-full-only: writing",
        threejs_obj.name,
        "(large file; several minutes possible)",
    )
    write_obj(
        geometries,
        uv_full,
        mat_mapping,
        threejs_obj,
        THREEJS_FULL_MTL_NAME,
        header_comment=(
            "Full-resolution mesh via trimesh (Three.js-compatible; "
            "Rhino NURBS appendix stripped)"
        ),
    )
    if MTL_SRC.exists():
        threejs_mtl_path = THREEJS_FULL_DIR / THREEJS_FULL_MTL_NAME
        shutil.copy2(MTL_SRC, threejs_mtl_path)
        log.info("Copied MTL to %s", threejs_mtl_path)
    texture_roots = [MODEL_OBJECTS_ROOT, OBJ_PATH.parent]
    print(f"[checkpoint] Copying referenced textures; roots={texture_roots}")
    copy_referenced_textures(MTL_SRC, THREEJS_FULL_DIR, texture_roots)

    json_path = OUT_DIR / "interaction_points.json"
    if not json_path.is_file():
        log.error("Missing %s — create it by running decimate_model.py once", json_path)
        sys.exit(1)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    model = data.setdefault("model", {})
    model.update(
        {
            "file": THREEJS_FULL_OBJ_NAME,
            "mtl_file": THREEJS_FULL_MTL_NAME,
            "mesh_base": VIEWER_THREEJS_MESH_BASE,
            "texture_path": VIEWER_THREEJS_TEX_PATH,
            "needs_center_offset": False,
            "center_offset": [float(c) for c in center],
            "original_faces": total_faces_orig,
        }
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Pointed viewer in %s at %s", json_path, THREEJS_FULL_OBJ_NAME)
    print(f"[checkpoint] Updated {json_path.name} - restart serve and hard-refresh viewer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decimate site OBJ or export Three.js-safe full mesh.",
    )
    parser.add_argument(
        "--export-threejs-full-only",
        action="store_true",
        help=(
            "Only write site_threejs_full.obj + MTL/textures and update "
            "interaction_points.json (skip decimation)."
        ),
    )
    args = parser.parse_args()
    print(f"[checkpoint] decimate_model argv={sys.argv}")
    if args.export_threejs_full_only:
        export_threejs_full_only()
    else:
        main()
