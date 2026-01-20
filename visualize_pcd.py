#!/usr/bin/env python3
import sys
import open3d as o3d
from pathlib import Path


def visualize(path):
    ext = path.suffix.lower()

    if ext == '.ply':
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            print(f"{path.name}: mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            o3d.visualization.draw_geometries([mesh], window_name=path.name, width=1024, height=768)
            return

    pcd = o3d.io.read_point_cloud(str(path))
    if not len(pcd.points):
        sys.exit(f"Failed to load: {path}")

    print(f"{path.name}: {len(pcd.points)} points, colors={pcd.has_colors()}, normals={pcd.has_normals()}")
    o3d.visualization.draw_geometries([pcd], window_name=path.name, width=1024, height=768,
                                       point_show_normal=pcd.has_normals())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <file.pcd|file.ply>")

    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"Not found: {path}")
    if path.suffix.lower() not in ['.pcd', '.ply']:
        sys.exit(f"Unsupported: {path.suffix}")

    visualize(path)
