# DriveVisionOcc
3D occupancy prediction from multi-camera images (based on DriveOcc, evaluated on the nuScenes occupancy task).  
The prediction and the ground truth are shown as colored voxel grids; each voxel is 0.4 m and colored by its semantic class.

# Demo
Two interactive replays are provided:

- `pred_results_replay.py` replays the stored predictions frame by frame in a single 3D window.
- `pred_results_replay_compare.py` shows the prediction and the ground truth side by side in two linked panes, so they can be compared directly.

In every GIF the 3D occupancy grid is on the left and the corresponding camera image is on the right.

## pred_results_replay.py - testset
<img src="gifs/replay_testset.gif" width="70%"/>

## pred_results_replay.py - trainset
<img src="gifs/replay_trainset.gif" width="70%"/>

## pred_results_replay_compare.py - testset
<img src="gifs/compare_testset.gif" width="70%"/>

## pred_results_replay_compare.py - trainset
<img src="gifs/compare_trainset.gif" width="70%"/>

# Usage
The replays read their data from `pred_dir_testset_already/` and `pred_dir_trainset_already/`.  
Each sample is three files:

```
{idx}_pred_occ.ply     # predicted occupancy voxels (x, y, z + rgb)
{idx}_gt_occ.ply       # ground-truth occupancy voxels
{idx}_cams_source.jpg  # the corresponding camera image
```

Run one of the replays:

```bash
python pred_results_replay.py
python pred_results_replay_compare.py
```

Controls:

- mouse in the 3D window: rotate (left), pan (middle / shift+left), zoom (wheel)
- keys (focus the OpenCV "Image Window"):

| key | action |
| --- | --- |
| `space` | pause / resume |
| `i` | reverse the playback direction |
| `e` | toggle the cube edges |
| `w` | toggle wireframe (edges only) |
| `g` | toggle the ground-truth overlay (`pred_results_replay.py` only) |
| `r` | reset the camera to the default viewpoint |
| `q` / `ESC` | quit |

# Make the demo GIFs
The four GIFs above are generated with `make_demo_gifs.py` (offscreen rendering, no window pops up; ~10 s each):

```bash
python make_demo_gifs.py                 # all four combinations
python make_demo_gifs.py replay testset  # a single combination
```

# Quickstart
The code has been tested on Ubuntu 22.04. An anaconda virtual env is recommended:

```bash
conda create -n drivevisionocc python=3.10
conda activate drivevisionocc
```

And install the dependencies:

```
pytorch >= 2.1
open3d >= 0.16.0
opencv-python >= 4.7
pyvista
pillow
```

# Inference
Put the model file `DriveVisionOcc.pt` in the root directory of the repository, then run:

```bash
python test.py
```

to run the inference and display the prediction as a voxel grid. The results (`*_pred_occ.ply`, `*_gt_occ.ply`, `*_cams_source.jpg`) are saved under `pred_dir/`.
