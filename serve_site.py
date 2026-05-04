"""Simple HTTP server for the 3D site viewer.

Serves all files from the project root on port 8080.
Run with:  uv run python serve_site.py

The 3D mesh URL is **not** configured here: ``site_viewer.html`` loads
``media/3D-objects/decimated/interaction_points.json``, which lists the OBJ/MTL
paths (e.g. full ``site_threejs_full.obj`` in new-3D-models or decimated).
"""

import http.server
import socketserver
import sys
from pathlib import Path

PORT = 8080


class SiteHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".stl": "application/octet-stream",
        ".obj": "application/octet-stream",
        ".mtl": "text/plain",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        # Avoid stale interaction_points.json / HTML while iterating locally.
        if self.path.endswith((".html", ".json")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    directory = str(Path(__file__).resolve().parent)
    handler = lambda *args, **kw: SiteHandler(*args, directory=directory, **kw)

    print("[checkpoint] serve_site: Cache-Control no-store for .html and .json")
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Serving at  http://localhost:{PORT}")
        print(f"Open viewer: http://localhost:{PORT}/site_viewer.html")
        print(f"Root dir:    {directory}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            sys.exit(0)


if __name__ == "__main__":
    main()
