r'''
v6: replay the prediction (left) and the ground truth (right) side by side in
    one window, with linked cameras: rotating / zooming one pane moves both,
    so the two voxel grids can be compared directly instead of overlaying them.
    The view is orthographic (see PARALLEL_PROJECTION): parallel lines stay
    parallel on screen, so the two panes read as aligned drawings.

Controls:
    mouse in the 3D window : rotate (left), pan (middle / shift+left), zoom (wheel)
    keys (focus the OpenCV "Image Window"):
        space : pause / resume
        i     : reverse playback direction
        e     : toggle cube edges
        w     : toggle wireframe (edges only)
        r     : reset the camera to the default viewpoint (behind and to the side of the ego, driving direction forward)
        q/ESC : quit
'''
import os
import math
import time
import gc

import numpy as np
import cv2
import pyvista as pv

base_root = './pred_dir_trainset_already/'

# --------------------------------------------------------------------------- #
# display parameters
# --------------------------------------------------------------------------- #
VOXEL_SIZE = 0.4        # m, prediction grid resolution
PARALLEL_PROJECTION = True  # orthographic view: no perspective convergence between the panes
CUBE_SHRINK = 0.9       # draw each cube slightly smaller so neighbouring voxels stay separated
EDGE_COLOR = 'black'    # color of the cube edges
EDGE_WIDTH = 1          # edge line width in pixels
BACKGROUND = 'black'
WINDOW_SIZE = (1600, 900)   # two viewports side by side (800x900 each)
CAM_IMG_SIZE = (640, 358)   # size of the OpenCV camera image window
PLAY_INTERVAL = 0.5         # s between frames while playing
INDEX_START = 0
INDEX_END = None            # None -> play all frames found in base_root

color_map = np.array(
                [
                    [255, 255, 255, 255],  # 0: background[0, 0, 0, 255],  # unknown
                    [160, 32, 240, 255],  #  2 truck  purple
                    [135, 60, 0, 255],  #  3 trailer  brown
                    [255, 255, 0, 255],  #  4 bus  yellow
                    [0, 255, 255, 255],  #  5 construction_vehicle  cyan
                    [255, 192, 203, 255],  #  6 bicycle  pink
                    [200, 180, 0, 255],  #  7 motorcycle   dark orange
                    [255, 0, 0, 255],  #  8 pedestrian  red
                    [255, 240, 150, 255],  #  9 traffic_cone   light yellow
                    [255, 120, 50, 255],  #  10 barrier  orangey
                    [255, 0, 255, 255],  # 11  driveable_surface  dark pink
                    [139, 137, 137, 255],  #  12 other_flat
                    [75, 0, 75, 255],  #  13  sidewalk
                    [150, 240, 80, 255],  # 14 terrain
                    [230, 230, 250, 255], # 15 manmade Burlywood
                    [0, 175, 0, 255],  # 16 vegetation  Green
                    [0, 150, 245, 255],  #  1 car  blue
                ], dtype=np.uint8
            )

# VTK hexahedron node order: v0..v3 bottom face, v4..v7 top face, aligned
_CUBE_OFFSETS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_voxel_mesh(xyz, labels=None, color_table=None, shrink=None, rgb=None):
    """One hexahedral cell per occupied voxel, colored by semantic label.

    Color each voxel either by `labels` (mapped through color_table) or with
    explicit per-voxel `rgb` values (uint8 0-255, e.g. read back from a saved
    ply). color_table / shrink default to this module's color_map / CUBE_SHRINK
    so other scripts can reuse the same rendering with their own palette.
    """
    if color_table is None:
        color_table = color_map
    if shrink is None:
        shrink = CUBE_SHRINK

    n = len(xyz)
    h = VOXEL_SIZE * shrink / 2.0
    corners = (xyz.astype(np.float32)[:, None, :] + _CUBE_OFFSETS[None] * h).reshape(-1, 3)

    cells = np.empty((n, 9), dtype=np.int64)
    cells[:, 0] = 8
    cells[:, 1:] = np.arange(8 * n, dtype=np.int64).reshape(n, 8)

    grid = pv.UnstructuredGrid(cells.ravel(),
                               np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8),
                               corners)
    if rgb is None:
        rgb = color_table[labels, :3]
    grid.cell_data['RGB'] = np.asarray(rgb)
    return grid


def load_voxel_mesh_from_ply(ply_file, shrink=None):
    """Load a point cloud ply (x, y, z + uchar rgb, as saved by open3d) and
    build the hexahedral voxel mesh, colored with the stored RGB values."""
    pcd = pv.read(ply_file)
    rgb = np.asarray(pcd.point_data['RGB'])
    if np.issubdtype(rgb.dtype, np.floating):   # some writers store 0..1 floats
        rgb = np.clip(rgb * 255.0, 0, 255)
    return build_voxel_mesh(np.asarray(pcd.points), rgb=rgb.astype(np.uint8),
                            shrink=shrink)


def add_voxel_actor(plotter, mesh, show_edges, wireframe):
    actor = plotter.add_mesh(mesh, rgb=True,
                             show_edges=show_edges,
                             edge_color=EDGE_COLOR,
                             line_width=EDGE_WIDTH,
                             smooth_shading=False)
    if wireframe:
        actor.prop.SetRepresentationToWireframe()
    return actor


def add_ego_axes(plotter, origin=(0.0, 0.0, 0.5), size=3.0):
    """Draw an RGB = XYZ coordinate frame at the ego vehicle position.

    The frame is lifted 0.5 m above the ground so the horizontal arrows are
    not buried inside the road-surface voxels around the ego.

    With scale='auto' the arrow length equals |direction| and the radii are
    scaled by the same factor, so the radii below are fractions of `size`.
    """
    origin = np.asarray(origin, dtype=np.float64)
    # explicit RGB tuples: pyvista resolves color names through matplotlib,
    # where 'green' is the half-bright #008000 instead of pure green
    for vec, color in [((1, 0, 0), (1.0, 0.0, 0.0)),
                       ((0, 1, 0), (0.0, 1.0, 0.0)),
                       ((0, 0, 1), (0.0, 0.0, 1.0))]:
        arrow = pv.Arrow(start=origin,
                         direction=np.asarray(vec, dtype=np.float64) * size,
                         tip_length=0.3, tip_radius=0.12,
                         shaft_radius=0.03, scale='auto')
        plotter.add_mesh(arrow, color=color, lighting=False)


def set_default_view(plotter, mesh=None, viewport_size=None):
    """Default viewpoint: behind and slightly to the side of the ego vehicle,
    looking along +x (the driving direction) with the same three-quarter
    angle as pred_results_replay.py, so the grid reads as a side view rather
    than a flat front elevation (z stays up).

    With PARALLEL_PROJECTION the camera is orthographic: parallel lines stay
    parallel on screen (no vanishing point), so the two side-by-side panes
    read as aligned drawings. Otherwise a 60deg vertical FOV perspective is
    used. With side-by-side subplots pass the size of one viewport (w, h),
    otherwise the whole window size is used.
    """
    if mesh is not None:
        bounds = np.asarray(mesh.bounds).reshape(3, 2)   # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    else:
        bounds = np.array([[-40.0, 40.0], [-40.0, 40.0], [-1.0, 5.0]])
    center = bounds.mean(1)
    w, h = plotter.window_size if viewport_size is None else viewport_size

    # from eye to scene center: behind, above and to the side of the road
    # (same direction as pred_results_replay.py)
    view_dir = np.array([0.60, 0.28, -0.36])
    view_dir = view_dir / np.linalg.norm(view_dir)

    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.focal_point = center
    plotter.camera.parallel_projection = PARALLEL_PROJECTION
    if PARALLEL_PROJECTION:
        # orthographic: fit the scene extent, parallel lines stay parallel.
        # the lateral view tilts the long horizontal scene extent into both
        # screen axes, so project the bounding-box corners onto the camera's
        # screen axes instead of fitting each world axis separately
        corners = np.array([[bounds[0, x], bounds[1, y], bounds[2, z]]
                            for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                           dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(view_dir, up)
        right /= np.linalg.norm(right)
        screen_up = up - np.dot(up, view_dir) * view_dir
        screen_up /= np.linalg.norm(screen_up)
        rel = corners - center
        proj_w = np.abs(rel @ right).max()
        proj_h = np.abs(rel @ screen_up).max()
        plotter.camera.parallel_scale = max(proj_h, proj_w * h / max(w, 1)) * 1.05
        # position only sets the viewpoint; framing comes from parallel_scale
        plotter.camera.position = center - view_dir * np.linalg.norm(bounds[:, 1] - bounds[:, 0])
    else:
        plotter.camera.view_angle = 60.0   # vertical FOV
        vfov = math.radians(60.0)
        hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * max(w / max(h, 1), 1.0))
        half_diag = 0.5 * np.linalg.norm(bounds[:, 1] - bounds[:, 0])
        distance = half_diag / min(math.tan(vfov / 2.0), math.tan(hfov / 2.0)) * 1.05
        plotter.camera.position = center - view_dir * distance
    plotter.camera.up = (0.0, 0.0, 1.0)


def discover_frames():
    frames = sorted(int(f.split('_')[0]) for f in os.listdir(base_root)
                    if f.endswith('_pred_occ.ply'))
    if INDEX_END is not None:
        frames = [f for f in frames if f < INDEX_END]
    return frames[INDEX_START:]


# --------------------------------------------------------------------------- #
# player
# --------------------------------------------------------------------------- #
def main():
    frames = discover_frames()
    if not frames:
        print('no *_pred_occ.ply found in {}'.format(base_root))
        return

    # one window, two viewports side by side: prediction (left) / ground truth
    # (right), with linked cameras so both panes always show the same viewpoint
    plotter = pv.Plotter(shape=(1, 2), window_size=WINDOW_SIZE, border=False)
    plotter.set_background(BACKGROUND)
    plotter.subplot(0, 0)
    # tuple position + viewport=True -> exact-size Text actor anchored to the
    # viewport's upper-left corner. Pass an explicit color: add_text otherwise
    # falls back to the GLOBAL theme font color, which is black (invisible on
    # the black background).
    plotter.add_text('Prediction', position=(0.02, 0.92), font_size=24,
                     viewport=True, color='white')
    add_ego_axes(plotter)
    plotter.subplot(0, 1)
    plotter.add_text('Ground Truth', position=(0.02, 0.92), font_size=24,
                     viewport=True, color='white')
    add_ego_axes(plotter)
    plotter.show(title='pred vs gt voxel grid', interactive_update=True, auto_close=False)
    plotter.link_views()

    pred_actor = None
    gt_actor = None
    pred_mesh = None
    gt_mesh = None
    show_edges = True
    wireframe = False
    paused = False
    inverse_operation = False
    pos = 0
    camera_fitted = False
    viewport_w = plotter.window_size[0] // 2
    viewport_h = plotter.window_size[1]

    def refresh_actors():
        nonlocal pred_actor, gt_actor
        if pred_actor is not None:
            plotter.remove_actor(pred_actor, render=False)
        plotter.subplot(0, 0)
        pred_actor = add_voxel_actor(plotter, pred_mesh, show_edges, wireframe)
        if gt_actor is not None:
            plotter.remove_actor(gt_actor, render=False)
            gt_actor = None
        plotter.subplot(0, 1)
        if gt_mesh is not None:
            gt_actor = add_voxel_actor(plotter, gt_mesh, show_edges, wireframe)

    while True:
        idx = frames[pos]
        print('sample {} is processing...'.format(idx))

        pred_file = os.path.join(base_root, '{}_pred_occ.ply'.format(idx))
        pred_mesh = load_voxel_mesh_from_ply(pred_file)

        gt_file = os.path.join(base_root, '{}_gt_occ.ply'.format(idx))
        gt_mesh = load_voxel_mesh_from_ply(gt_file) if os.path.exists(gt_file) else None

        # quick sanity check: the right pane must load the GT, not the pred
        print('  pred {} cells | gt {} cells'.format(pred_mesh.n_cells,
               gt_mesh.n_cells if gt_mesh is not None else 'MISSING'))

        refresh_actors()
        if not camera_fitted:
            # both panes share one camera: fit it once to the pred scene
            set_default_view(plotter, pred_mesh, viewport_size=(viewport_w, viewport_h))
            camera_fitted = True
        try:
            plotter.update()
        except RuntimeError:
            break  # the 3D window was closed

        img_save_path = os.path.join(base_root, '{}_cams_source.jpg'.format(idx))
        img_source = cv2.imread(img_save_path)
        img_shown = False
        if img_source is not None:
            img_source = cv2.resize(img_source, CAM_IMG_SIZE)
            cv2.imshow('Image Window', img_source)
            img_shown = True

        # --- input -------------------------------------------------------- #
        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):
            paused = not paused
            print('display paused: {}'.format(paused))
        elif key == ord('i'):
            inverse_operation = not inverse_operation
            print('display inverse: {}'.format(inverse_operation))
        elif key == ord('e'):
            show_edges = not show_edges
            print('show edges: {}'.format(show_edges))
            refresh_actors()
        elif key == ord('w'):
            wireframe = not wireframe
            print('wireframe: {}'.format(wireframe))
            refresh_actors()
        elif key == ord('r'):
            set_default_view(plotter, pred_mesh, viewport_size=(viewport_w, viewport_h))
            print('camera reset')
        elif key in (ord('q'), 27):  # q / ESC
            break

        # keep the 3D window responsive while paused
        if paused or key in (ord('e'), ord('w')):
            plotter.update()

        # --- advance ------------------------------------------------------ #
        if not paused:
            if not inverse_operation:
                pos += 1
                if pos >= len(frames):
                    pos = len(frames) - 1
            else:
                pos -= 1
                if pos < 0:
                    pos = 0
            time.sleep(PLAY_INTERVAL)

        # quit when the user closes one of the windows
        if (img_shown and cv2.getWindowProperty('Image Window', cv2.WND_PROP_VISIBLE) < 1
                or plotter.render_window is None):
            break

    print('done')
    plotter.close()
    cv2.destroyAllWindows()

    # release the pyvista objects while the VTK modules are still alive,
    # otherwise their __del__ cleanup may run during interpreter shutdown
    # and print "Exception ignored ..." errors
    del plotter, pred_actor, gt_actor, pred_mesh, gt_mesh
    gc.collect()


if __name__ == '__main__':
    main()
