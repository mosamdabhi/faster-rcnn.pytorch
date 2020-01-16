from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
from scipy.misc import imread
import random
import matplotlib.pyplot as plt
def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
freihand_data = "/home/mqadri/hand-integral-pose-estimation/data"
add_pypath(freihand_data)
freihand_common = "/home/mqadri/hand-integral-pose-estimation/common"
add_pypath(freihand_common)
freihand_main = "/home/mqadri/hand-integral-pose-estimation/main"
add_pypath(freihand_main)

plt.switch_backend('agg')
import config as cfg
from FreiHand_config import FreiHandConfig
from FreiHand import FreiHand
import augment

def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            im = cv2.rectangle(im, bbox[0:2], bbox[2:4], (255, 0, 0), 2)
            im = cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


patch_width = 224
patch_height = 224

def calc_kpt_bound(kpts, kpts_vis):
    MAX_COORD = 10000
    x = kpts[:, 0]
    y = kpts[:, 1]
    z = kpts_vis[:, 0]
    u = MAX_COORD
    d = -1
    l = MAX_COORD
    r = -1
    for idx, vis in enumerate(z):
        if vis == 0:  # skip invisible joint
            continue
        u = min(u, y[idx])
        d = max(d, y[idx])
        l = min(l, x[idx])
        r = max(r, x[idx])
    l = np.clip(l, 0, patch_width - 1)
    r = np.clip(r, 0, patch_width - 1)
    u = np.clip(u, 0, patch_height - 1)
    d = np.clip(d, 0, patch_height - 1)
    return u, d, l, r

def find_bb(uv, joint_vis, aspect_ratio=1.0):
    u, d, l, r = calc_kpt_bound(uv, joint_vis)

    center_x = (l + r) * 0.5
    center_y = (u + d) * 0.5
    assert center_x >= 1

    w = r - l
    h = d - u
    assert w > 0
    assert h > 0
    #bbox = xywh_to_xyxy(np.array([center_x, center_y, w, h]))

    #if w > aspect_ratio * h:
    #    h = w * 1.0 / aspect_ratio
    #elif w < aspect_ratio * h:
    #    w = h * aspect_ratio
    
    #w *= 1.75
    #h *= 1.75
    
#===============================================================================
    if center_x + w/2 >= 224:
        print("1")
        print(r, l, d, u)
        print(center_x + w/2)
    if center_x - w/2 < 0:
        print("2")
        print(r, l, d, u)
        print(center_x - w/2)
        #w = min(2*center_x - 1, w)
 
    if center_y + h/2 >= 224:
        print("3")
        print(r, l, d, u)
        print(center_y + h/2)
    if center_y - h/2 < 0:
        print("4")
        print(r, l, d, u)
        print(center_y - h/2)
#    
    x1 = center_x - w*0.5
    x2 = center_x + w*0.5
    y1 = center_y - h*0.5
    y2 = center_y + h*0.5
#     if x1 > 224 or x1 < 0 or x2 > 224 or x2 < 0 or y1 > 225 or y1 < 0 or y2 > 225 or y2 < 0:
#         print(x1)
#         print(x2)
#         print(y1)
#         print(y2)
#         sys.exit()
#===============================================================================
    #w = np.clip(w, 0, cfg.patch_width)
    #h = np.clip(h, 0, cfg.patch_height)
    #bbox = [center_x, center_y, w, h]
    bbox = [x1, y1, x2, y2]
    print(bbox)
    return bbox

if __name__ == "__main__":
    data_split = "testing"
    f = FreiHand(data_split)
    f.data_dir = "/home/mqadri/hand-integral-pose-estimation/data/FreiHand"
    data = f.load_data()
    bboxes = []
    
    # load 100 testing samples for testing
    i = 0
    if data_split in ["training", "testing"]:
        dir_name = "training"
    else:
        dir_name = data_split
    for d in data:
        i += 1
        if i > 2:
            break
        joint_cam = d["joint_cam"]
        K = d['K']
        joint_vis = np.ones(joint_cam.shape, dtype=np.float)
        uv, _, _= augment.projectPoints(joint_cam, np.eye(3), K)
        img_name = d['img_path'].split('/')[-1]
        bbox_info = {
            'bbox' : find_bb(uv, joint_vis),
            #'idx': d['idx'],
            'img_path': '/home/mqadri/faster-rcnn.pytorch/data/Freihand/{}/rgb/{}'.format(dir_name, img_name),
            'class': 'hand'
        }
        bboxes.append(bbox_info)
        im_in = np.array(imread(bbox_info["img_path"]))
        if len(im_in.shape) == 2:
            im_in = im_in[:,:,np.newaxis]
            im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        # rgb -> bgr
        im = im_in[:,:,::-1]
        cvimg = cv2.imread(bbox_info["img_path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        joint_cam = d['joint_cam']
        K = d['K']
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        #            
        #     ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #     # 
        #     ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
        ax1.imshow(cvimg)
        #     #ax1.imshow(img2_w)
        #     # 
        uv, _, _ = augment.projectPoints(joint_cam, np.eye(3), K)
        f.plot_hand(ax1, uv, order='uv')
        #ax1.imshow(cvimg)
        #     FreiHand.plot_hand(ax2, joint_img_orig[:, 0:2], order='uv')
        #     ax1.axis('off')
        nn = str(random.randint(1,999))
        #     #print("=============================================================")
        #     #print(nn)
                          


     
        pascal_classes = ["hand"]
        bbox = bbox_info["bbox"]
        bbox = np.array(bbox)
        bbox = np.expand_dims(bbox, 0)
        #bbox = xywh_to_xyxy(bbox)
        ones = np.ones(bbox.shape[0])
        ones = np.expand_dims(ones, 0)
        bbox = np.hstack((bbox, ones))
        #for i in range(bbox.shape[0]):
        #   print(type(bbox[i]))
        #   bbox[i].append(1) 
        im2show = vis_detections(im, pascal_classes[0], bbox, 0.5)
        im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        print(type(im2showRGB))
        peo = im2showRGB.get().astype('f')
        ax2.imshow((255*peo/np.max(peo)).astype(np.uint8))
        plt.savefig('/home/mqadri/faster-rcnn.pytorch/tests/{}.jpg'.format(nn))
        #nn = str(random.randint(1000,2000))
        #cv2.imwrite('/home/mqadri/faster-rcnn.pytorch/tests/{}.jpg'.format(nn), im2showRGB)
        
    np.save("freihand_bbox_gt_{}".format(data_split), bboxes)
