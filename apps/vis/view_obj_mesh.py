import argparse
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=str, help="Path to an .obj mesh file")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--axis", type=float, default=0.3, help="Coordinate axis size")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.obj)
    if mesh.is_empty():
        raise RuntimeError("Failed to load mesh: {}".format(args.obj))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.75, 0.75, 0.92])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SMPL Mesh", width=args.width, height=args.height)
    vis.add_geometry(mesh)
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = [0.9, 0.9, 0.9]
    opt.mesh_show_back_face = True
    opt.light_on = True

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

