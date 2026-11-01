r'''
test with voxel grid: run DriveVisionOcc inference and display the prediction as a
voxel grid (cubes with visible edges) instead of a point cloud.
The voxel rendering is shared with npy_vis_dir_voxel_grid.py and runs on the
GPU via pyvista (VTK / OpenGL).

Controls:
    mouse in the 3D window : rotate (left), pan (middle / shift+left), zoom (wheel)
    keys (focus the OpenCV "Image Window"):
        space : pause / continue to the next sample
        e     : toggle cube edges
        w     : toggle wireframe (edges only)
        g     : toggle the ground-truth voxel overlay (drawn smaller, inside the pred)
        r     : reset the camera to the default viewpoint (behind the ego, driving direction forward)
        q/ESC : quit
'''
import torch
from utils.utils import DriveVisionOccDataset, color_map, post_process, gt_occ_lable_to_point_cloud
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import time
import cv2
import gc
import pyvista as pv

# shared voxel rendering (same look as pred_results_replay.py)
from pred_results_replay import build_voxel_mesh, add_voxel_actor, set_default_view, add_ego_axes

WINDOW_SIZE = (1200, 900)
BACKGROUND = 'black'
CAM_IMG_SIZE = (640, 241)
GT_SHRINK = 0.75    # gt cubes are drawn smaller so they nest inside the pred cubes

def show_multi_imgs(scale, imglist, order=None, border=10, border_color=(255, 255, 0)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws , hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = allimgs[i * order[1] + j]
    return imgblank

if __name__=="__main__":
    dataset_dir = './data/nuscenes_infos_test_occ_mini_set.pkl'
    test_dataset = DriveVisionOccDataset(dataset_dir)
    collate_fn = lambda x:x
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, collate_fn = collate_fn, prefetch_factor=1)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    # load the scripted (TorchScript) model: no eager DriveVisionOcc / checkpoint needed
    drivevisionocc = torch.jit.load('./DriveVisionOcc.pt')
    drivevisionocc.to(device).eval()

    #voxel display (pyvista / VTK, GPU accelerated), opened once the model is ready
    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    plotter.set_background(BACKGROUND)
    add_ego_axes(plotter)
    plotter.show(title='occ voxel grid', interactive_update=True, auto_close=False)

    pred_actor = None
    gt_actor = None
    pred_mesh = None
    gt_mesh = None
    show_gt = False
    show_edges = True
    wireframe = False
    paused = False
    quit_requested = False
    to_reset = True

    def refresh_actors():
        global pred_actor, gt_actor
        if pred_actor is not None:
            plotter.remove_actor(pred_actor)
            pred_actor = None
        if gt_actor is not None:
            plotter.remove_actor(gt_actor)
            gt_actor = None
        if pred_mesh is not None:
            pred_actor = add_voxel_actor(plotter, pred_mesh, show_edges, wireframe)
        if show_gt and gt_mesh is not None:
            gt_actor = add_voxel_actor(plotter, gt_mesh, show_edges, wireframe)

    for sample_idx, data in enumerate(test_dataloader):
        bs = len(data)
        img_data = torch.cat([torch.from_numpy(data[i]['img']).unsqueeze(0) for i in range(bs)],dim=0).to(device)

        with torch.no_grad():
            pred_occ = drivevisionocc(img_data)#inferrence

            ###########################################
            #post process and display and save
            ###########################################
            pred_occ_array = post_process(pred_occ)

            #voxel grid display
            pred_labels = pred_occ_array[:, 3].astype(int)
            pred_mesh = build_voxel_mesh(pred_occ_array[:, :3], pred_labels, color_map)
            refresh_actors()
            if to_reset:
                set_default_view(plotter, pred_mesh)
                to_reset = False
            try:
                plotter.update()
            except RuntimeError:
                break  # the 3D window was closed

            #pcd save
            pred_occ_pcd = o3d.geometry.PointCloud()
            pred_occ_pcd.points = o3d.utility.Vector3dVector(pred_occ_array[:, :3])
            color = color_map[pred_labels] / 255.0
            pred_occ_pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
            pred_file_path = 'pred_dir/' + str(sample_idx) + '_pred_occ.ply'
            o3d.io.write_point_cloud(pred_file_path, pred_occ_pcd) #save

            #################################
            #image read and display and save
            #################################
            img_data = data[0]['img']
            frame1 = np.transpose(img_data[0],axes=(2,1,0))#Front
            frame2 = np.transpose(img_data[1],axes=(2,1,0))#Front Right
            frame3 = np.transpose(img_data[2],axes=(2,1,0))#Front Left
            frame4 = np.transpose(img_data[3],axes=(2,1,0))#Back
            frame5 = np.transpose(img_data[4],axes=(2,1,0))#Back Left
            frame6 = np.transpose(img_data[5],axes=(2,1,0))#Back Right
            img_source = show_multi_imgs(0.4, [frame3, frame1, frame2, frame5, frame4, frame6], (2, 3))
            img_source = cv2.resize(img_source, CAM_IMG_SIZE)
            img_save_path = 'pred_dir/' + str(sample_idx) + '_cams_source.jpg'
            cv2.imwrite(img_save_path,img_source)
            cv2.imshow('Image Window',img_source)

            #########################
            #occ gt save for compare
            #########################
            occ_gt_label = data[0]['gt_occ']
            gt_occ_array = gt_occ_lable_to_point_cloud(occ_gt_label)
            gt_labels = occ_gt_label[occ_gt_label != 0].astype(int)

            #gt voxel overlay (optional, key 'g'), drawn smaller so it nests inside the pred
            if show_gt:
                gt_mesh = build_voxel_mesh(gt_occ_array, gt_labels, color_map, shrink=GT_SHRINK)
                refresh_actors()

            gt_occ_pcd = o3d.geometry.PointCloud()
            gt_occ_pcd.points = o3d.utility.Vector3dVector(gt_occ_array)
            #color rendering
            color = color_map[gt_labels] / 255.0
            gt_occ_pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
            #pcd save
            pred_file_path = 'pred_dir/' + str(sample_idx) + '_gt_occ.ply'
            o3d.io.write_point_cloud(pred_file_path, gt_occ_pcd) #save

            print('sample {} is ok..'.format(sample_idx))

            # --- input -------------------------------------------------------- #
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' '):
                paused = not paused
                print('display paused: {}'.format(paused))
            elif key == ord('e'):
                show_edges = not show_edges
                print('show edges: {}'.format(show_edges))
                refresh_actors()
            elif key == ord('w'):
                wireframe = not wireframe
                print('wireframe: {}'.format(wireframe))
                refresh_actors()
            elif key == ord('g'):
                show_gt = not show_gt
                print('show gt: {}'.format(show_gt))
                if show_gt:
                    gt_mesh = build_voxel_mesh(gt_occ_array, gt_labels, color_map, shrink=GT_SHRINK)
                else:
                    gt_mesh = None
                refresh_actors()
            elif key == ord('r'):
                set_default_view(plotter, pred_mesh)
                print('camera reset')
            elif key in (ord('q'), 27):  # q / ESC
                break

            # keep the 3D window responsive while paused
            while paused:
                cv2.imshow('Image Window',img_source)
                try:
                    plotter.update()
                except RuntimeError:
                    quit_requested = True
                    paused = False
                    break  # the 3D window was closed
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):
                    paused = False
                    print('display restored!')
                elif key in (ord('q'), 27):
                    quit_requested = True
                    paused = False
                    break
                if cv2.getWindowProperty('Image Window', cv2.WND_PROP_VISIBLE) < 1:
                    quit_requested = True
                    paused = False
                    break

            if quit_requested:
                break

            time.sleep(1)


    print('ok!')
    plotter.close()
    cv2.destroyAllWindows()

    # release the pyvista objects while the VTK modules are still alive,
    # so their __del__ cleanup runs now instead of during interpreter
    # shutdown (where it prints "Exception ignored ..." errors)
    del plotter, pred_actor, gt_actor, pred_mesh, gt_mesh
    gc.collect()
