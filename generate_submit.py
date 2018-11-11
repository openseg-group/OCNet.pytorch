##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import argparse
import scipy
from scipy import ndimage
import torch, cv2
import numpy as np
import sys
import pdb

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from collections import OrderedDict
from dataset import get_segmentation_dataset
from network import get_segmentation_model
from config import Parameters
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage

import matplotlib.pyplot as plt
import torch.nn as nn
torch_ver = torch.__version__[:3]


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, tile_size, classes, method, scale=1):
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img = image

    N_, C_, H_, W_ = scaled_img.shape

    if torch_ver == '0.4':
        interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=tile_size, mode='bilinear')

    full_probs = np.zeros((N_, H_, W_, classes))
    count_predictions = np.zeros((N_, H_, W_, classes))
    overlap = 0
    stride_h = ceil(tile_size[0] * (1 - overlap))
    stride_w = ceil(tile_size[1] * (1 - overlap))
    tile_rows = int(ceil((H_ - tile_size[0]) / stride_h) + 1)  # strided convolution formula
    tile_cols = int(ceil((W_ - tile_size[1]) / stride_w) + 1)
    print("Need %i x %i prediction tiles @ stride %i px, %i py" % (tile_cols, tile_rows, stride_h, stride_w))

    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride_w)
            y1 = int(row * stride_h)
            x2 = min(x1 + tile_size[1], W_)
            y2 = min(y1 + tile_size[0], H_)
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = scaled_img[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction_ = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda(), )
    
            if 'dsn' in method or 'center' in method or 'fuse' in method:
                padded_prediction = padded_prediction_[-1]
            else:
                padded_prediction = padded_prediction_

            padded_prediction = F.upsample(input=padded_prediction, size=tile_size, mode='bilinear', align_corners=True)
            padded_prediction = padded_prediction.cpu().data.numpy().transpose(0,2,3,1)                
            # padded_prediction = interp(padded_prediction).cpu().data.numpy().transpose(0,2,3,1)
            prediction = padded_prediction[:, 0:img.shape[2], 0:img.shape[3], :]
            count_predictions[:, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction 

    full_probs /= count_predictions
    full_probs = ndimage.zoom(full_probs, (1., 1./scale, 1./scale, 1.),
        order=1, prefilter=False)
    return full_probs


def predict_whole_img(net, image, classes, method, scale):
    """
         Predict the whole image w/o using multiple crops.
         The scale specify whether rescale the input image before predicting the results.
    """
    N_, C_, H_, W_ = image.shape

    if torch_ver == '0.4':
        interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(H_, W_), mode='bilinear')

#     bug
#     if scale > 1:
#         scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
#     else:
#         scaled_img = image

    scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    
    full_prediction_ = net(Variable(torch.from_numpy(scaled_img), volatile=True).cuda(), )
    if 'dsn' in method:
        full_prediction = full_prediction_[-1]
    else:
        full_prediction = full_prediction_

    if torch_ver == '0.4':
        full_prediction = F.upsample(input=full_prediction, size=(H_, W_), mode='bilinear', align_corners=True)
    else:
        full_prediction = F.upsample(input=full_prediction, size=(H_, W_), mode='bilinear')
    
    result = full_prediction.cpu().data.numpy().transpose(0,2,3,1)

    # result = interp(full_prediction).cpu().data.numpy().transpose(0,2,3,1)
    return result


def predict_multi_scale(net, image, scales, classes, flip_evaluation, method):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))  
    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        sys.stdout.flush()
        if scale <= 1.0:
            scaled_probs = predict_whole_img(net, image, classes, method, scale=scale)
        else:        
            scaled_probs = predict_sliding(net, image, (1024,2048), classes, method, scale=scale)
        if flip_evaluation == 'True':
            if scale <= 1.0:
                flip_scaled_probs = predict_whole_img(net, image[:,:,:,::-1].copy(), classes, method, scale=scale)
            else:
                flip_scaled_probs = predict_sliding(net, image[:,:,:,::-1].copy(), (1024,2048), classes, method, scale=scale)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,:,::-1])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs


def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


def main():
    """Create the model and start the evaluation process."""
    args = Parameters().parse()

    # file_log = open(args.log_file, "w")
    # sys.stdout = sys.stderr = file_log

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    sys.stdout.flush()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    ignore_label= args.ignore_label
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    deeplab = get_segmentation_model("_".join([args.network, args.method]), num_classes=args.num_classes)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    saved_state_dict = torch.load(args.restore_from)
    deeplab.load_state_dict(saved_state_dict)

    model = nn.DataParallel(deeplab)
    model.eval()
    model.cuda()

    testloader = data.DataLoader(get_segmentation_dataset(args.dataset, root=args.data_dir, list_path=args.data_list, 
                                    crop_size=(1024, 2048), scale=False, mirror=False, network=args.network),
                                    batch_size=args.batch_size, shuffle=False, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))
    palette = get_palette(256)

    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                  7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                  18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                  28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    image_id = 0
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, size, name = batch
        size = size[0].numpy()

        if torch_ver == '0.3':
            if args.use_ms == 'True': 
                output = predict_multi_scale(model, image.numpy(), ([0.75, 1, 1.25]), 
                    args.num_classes, args.use_flip, args.method)
            else:
                output = predict_whole_img(model, image.numpy(), args.num_classes, 
                    args.method, scale=float(args.whole_scale))
        else:
            with torch.no_grad():
                if args.use_ms == 'True': 
                    output = predict_multi_scale(model, image.numpy(), ([0.75, 1, 1.25]), 
                        args.num_classes, args.use_flip, args.method)
                else:
                    output = predict_whole_img(model, image.numpy(), args.num_classes, 
                        args.method, scale=float(args.whole_scale))


        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        seg_pred = id2trainId(seg_pred, id_to_trainid, reverse=True)

        for i in range(image.size(0)): 
            image_id += 1
            print('%d th segmentation map generated ...'%(image_id))
            sys.stdout.flush()
            if args.store_output == 'True':
                output_im = PILImage.fromarray(seg_pred[i])
                output_im.putpalette(palette)
                output_im.save(output_path+'/'+name[i]+'.png')

if __name__ == '__main__':
    main()
