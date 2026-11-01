import numpy as np
from numpy import random
import torch
import os
import torch.utils
import torch.utils.data
import pickle
import cv2

class MultiViewImageResize(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, img_scale=None,backend=None):
        self.img_scale = img_scale
        self.backend = backend

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        img_num = len(results['img'])
        for i in range(img_num):
            img_to_resize = results['img'][i]
            img_resized = cv2.resize(img_to_resize,
                                     self.img_scale,
                                     interpolation=cv2.INTER_LINEAR )
            results['img'][i] = img_resized
        h,w,c = results['img'][0].shape
        results['img_shape']=h,w,c,img_num
        results['pad_shape']=h,w,c,img_num
        return results

class DriveVisionOccDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_dir):
        super().__init__()
        self.use_semantic = True
        self.img_resize = MultiViewImageResize((800,448),'cv2')

        self.dir = dataset_dir
        # 打开一个文件用于写入
        with open(dataset_dir, 'rb') as f:
            # 使用pickle.dump()将字典对象序列化并保存到文件中
            try:
                self.data_info = pickle.load(f)
            except EOFError:
                return None
    
    def __len__(self):
        return len(self.data_info['infos'])
    
    def __getitem__(self, index):
        data_item_info = self.data_info['infos'][index]
        cam_files = data_item_info['camera']
        occ_file = data_item_info['occ_path']

        result = dict()
        #=================
        #image process
        #=================
        img_list = []
        img_read_rst = dict()
        for cam_name,file_path in cam_files.items():
                img = cv2.imread(file_path['data_path']).astype(np.float32)
                img_list.append(img)
        img_read_rst['img'] = img_list
        img_rst = self.img_resize(img_read_rst)
        img = np.stack(img_rst['img'],axis=0).astype(np.float32)
        img = img.transpose(0,3,2,1)#!opencv
        result['img'] = img

        #=================
        #occ gt process
        #=================
        occ = np.load(occ_file)
        occ = occ['semantics'].astype(np.float32)
        occ[occ==0] = 18
        occ[occ==16] = 0
        occ[occ==18] = 16
        gt_occ = np.zeros([240, 240, 24])
        gt_occ[:200,:200,:16] = occ
        result['gt_occ'] = gt_occ

        return result

color_map = np.array(
                [   [0, 0, 0, 255],  # unknown
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
                ]
            )

def post_process(pred_occ):
    filter_1 = torch.sigmoid(pred_occ['binary_pred'][:,0,...])#!
    class_num = pred_occ['shape_and_semantic_pred'][0].shape[1]

    conf1 = 0.6
    conf2 = 0.6
    for i in range(pred_occ['binary_pred'].shape[0]):
        filter_1 = filter_1 > conf1

        occ_before_upsample_reshape = []
        for n in range(27):
            occ_before_upsample_reshape.append( torch.zeros([1,class_num,80, 80, 8]).to(torch.device(pred_occ['binary_pred'].device)).type(torch.float) .permute(0,2,3,4,1))

        pred = pred_occ['shape_and_semantic_pred']
        for n in range(27):
            pred_item = pred[n]
            pred_item = pred_item.permute(0,2,3,4,1)
            occ_before_upsample_reshape [n][filter_1] = pred_item[filter_1]
            occ_before_upsample_reshape[n] =occ_before_upsample_reshape[n].permute(0,4,1,2,3)

        occ_size = [240, 240, 24]
        occ_result = torch.zeros([1,class_num, occ_size[0], occ_size[1], occ_size[2]]).to(pred[0].device).type(torch.float) #!
        #----------------------------111----------------------------------#
        occ_result[:,:,0:occ_size[0]:3,0:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[0]
        occ_result[:,:,1:occ_size[0]:3,0:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[1]
        occ_result[:,:,2:occ_size[0]:3,0:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[2]

        occ_result[:,:,0:occ_size[0]:3,1:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[3]
        occ_result[:,:,1:occ_size[0]:3,1:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[4]
        occ_result[:,:,2:occ_size[0]:3,1:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[5]

        occ_result[:,:,0:occ_size[0]:3,2:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[6]
        occ_result[:,:,1:occ_size[0]:3,2:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[7]
        occ_result[:,:,2:occ_size[0]:3,2:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[8]
        #----------------------------222----------------------------------#
        occ_result[:,:,0:occ_size[0]:3,0:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[9]
        occ_result[:,:,1:occ_size[0]:3,0:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[10]
        occ_result[:,:,2:occ_size[0]:3,0:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[11]

        occ_result[:,:,0:occ_size[0]:3,1:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[12]
        occ_result[:,:,1:occ_size[0]:3,1:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[13]
        occ_result[:,:,2:occ_size[0]:3,1:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[14]

        occ_result[:,:,0:occ_size[0]:3,2:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[15]
        occ_result[:,:,1:occ_size[0]:3,2:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[16]
        occ_result[:,:,2:occ_size[0]:3,2:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[17]
        #----------------------------333----------------------------------#
        occ_result[:,:,0:occ_size[0]:3,0:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[18]
        occ_result[:,:,1:occ_size[0]:3,0:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[19]
        occ_result[:,:,2:occ_size[0]:3,0:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[20]

        occ_result[:,:,0:occ_size[0]:3,1:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[21]
        occ_result[:,:,1:occ_size[0]:3,1:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[22]
        occ_result[:,:,2:occ_size[0]:3,1:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[23]

        occ_result[:,:,0:occ_size[0]:3,2:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[24]
        occ_result[:,:,1:occ_size[0]:3,2:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[25]
        occ_result[:,:,2:occ_size[0]:3,2:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[26]
        
        tensor_2_equal = torch.zeros([1,class_num, 1,1,1]).to(pred[0].device).type(torch.float) #!
        occ_result_filter = torch.eq(occ_result,tensor_2_equal)
        occ_result_filter = ~occ_result_filter
        occ_result_filter = torch.sum(occ_result_filter,dim=1,keepdim=False)
        occ_result_filter = occ_result_filter >0

        occ_result = occ_result.permute(0,2,3,4,1)

        x = torch.linspace(0, 239, occ_size[0])
        y = torch.linspace(0, 239, occ_size[1])
        z = torch.linspace(0, 23, occ_size[2])
        X, Y, Z = torch.meshgrid(x, y, z)
        vv = torch.stack([X, Y, Z], dim=-1).to(occ_result.device).unsqueeze(0)
    
        occ_result = occ_result[occ_result_filter]
        occ_result_binary = occ_result[:,0]
        occ_result_semantic = occ_result[:,1:]

        occ_result_binary = occ_result_binary.sigmoid()
        filter_2 = occ_result_binary < (1-conf2)
        occ_result_semantic = occ_result_semantic[filter_2]

        conf , j = occ_result_semantic.max(1,keepdim=True)
        j += 1
        vertices = vv[occ_result_filter]
        vertices = vertices[filter_2]
        vertices = torch.cat([vertices,j],dim=1)

        occ_range = [-40.0,-40.0,-1.0,56.0,56.0,8.6]
        vertices[:, 0] = (vertices[:, 0] + 0.5) * (occ_range[3] - occ_range[0]) / occ_size[0] + occ_range[0]
        vertices[:, 1] = (vertices[:, 1] + 0.5) * (occ_range[4] - occ_range[1]) / occ_size[1] + occ_range[1]
        vertices[:, 2] = (vertices[:, 2] + 0.5) * (occ_range[5] - occ_range[2]) / occ_size[2] + occ_range[2]
        vertices = vertices.cpu().numpy()

    return vertices

def gt_occ_lable_to_point_cloud(occ_label):
    x = torch.linspace(0, 239, 240)
    y = torch.linspace(0, 239, 240)
    z = torch.linspace(0, 23, 24)
    X, Y, Z = torch.meshgrid(x, y, z)
    gt_occ = torch.stack([X, Y, Z], dim=-1).to(occ_label.device)
    gt_occ = gt_occ[occ_label!=0]
    gt_occ = gt_occ.cpu().numpy()
    occ_range = [-40.0,-40.0,-1.0,56.0,56.0,8.6]
    occ_size = [240, 240, 24]
    gt_occ[:, 0] = (gt_occ[:, 0] + 0.5) * (occ_range[3] - occ_range[0]) /  occ_size[0]  + occ_range[0]
    gt_occ[:, 1] = (gt_occ[:, 1] + 0.5) * (occ_range[4] - occ_range[1]) /  occ_size[1]  + occ_range[1]
    gt_occ[:, 2] = (gt_occ[:, 2] + 0.5) * (occ_range[5] - occ_range[2]) /  occ_size[2]  + occ_range[2]

    return gt_occ
    
   

