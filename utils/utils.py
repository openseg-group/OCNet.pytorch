from PIL import Image
import numpy as np
import torch
import pdb

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def decode_predictions(preds, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(preds, list):
        preds_list = []
        for pred in preds:
            preds_list.append(pred[-1].data.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
    else:
        preds = preds.data.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    n, h, w = preds.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
      pixels = img.load()
      for j_, j in enumerate(preds[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    imgs = imgs.data.cpu().numpy()
    n, c, h, w = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (np.transpose(imgs[i], (1,2,0)) + img_mean).astype(np.uint8)
    return outputs


def reshape_predict_target(predict, target):
    # get all the valid prediction vectors and label vectors.
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
    assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
    assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
    n, c, h, w = predict.size()

    ntarget = target.data.cpu().numpy()
    cls_ids = np.unique(ntarget)

    # get all the valid class label list
    cls_ids = cls_ids[cls_ids != 255]
    cls_ids = [cls_id for cls_id in cls_ids if np.sum(ntarget == cls_id) > 20]
    # cls_ids = torch.from_numpy(np.array(cls_ids))

    input_feature = predict.view(-1, c)
    input_label = target.view(-1)

    index = [i for i in range(input_label.size(0)) if input_label[i].data.cpu().numpy() in cls_ids]

    input_feature = input_feature[index, :]
    input_label = input_label[index]

    return input_feature, input_label


# def down_sample_target(target, input_scale, output_scale):
def down_sample_target(target, scale):
    # pdb.set_trace()
    n, row, col = target.size(0), target.size(1), target.size(2)
    step = scale
    r_target = target[:, 0:row:step, :]
    # r_target = torch.cat((r_target, target[:, row-1, :]), 1) #here deal with the special case, need to add one more row and one more col    
    c_target = r_target[:, :, 0:col:step]
    # c_target = torch.cat((c_target, r_target[:, :, col-1]), 2)
    return c_target

def _quick_countless(data):
    """
    Vectorized implementation of downsampling a 2D 
    image by 2 on each side using the COUNTLESS algorithm.
  
    data is a 2D numpy array with even dimensions.
    """
    sections = []
    
    # This loop splits the 2D array apart into four arrays that are
    # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
    # and (1,1) representing the A, B, C, and D positions from Figure 1.
    factor = (2,2)
    for offset in np.ndindex(factor):
      part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
      sections.append(part)

    a, b, c, d = sections

    ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
    bc = b * (b == c) # PICK(B,C)

    a = ab_ac | bc # (PICK(A,B) || PICK(A,C)) or PICK(B,C)
    return a + (a == 0) * d # AB || AC || BC || D

def _zero_corrected_countless(data):
    """
    Vectorized implementation of downsampling a 2D 
    image by 2 on each side using the COUNTLESS algorithm.
    
    data is a 2D numpy array with even dimensions.
    """
    # allows us to prevent losing 1/2 a bit of information 
    # at the top end by using a bigger type. Without this 255 is handled incorrectly.
    # data, upgraded = upgrade_type(data) 
    # offset from zero, raw countless doesn't handle 0 correctly
    # we'll remove the extra 1 at the end.
    data += 1 

    sections = []
    
    # This loop splits the 2D array apart into four arrays that are
    # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
    # and (1,1) representing the A, B, C, and D positions from Figure 1.
    factor = (2,2)
    for offset in np.ndindex(factor):
      part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
      sections.append(part)

    a, b, c, d = sections

    ab = a * (a == b) # PICK(A,B)
    ac = a * (a == c) # PICK(A,C)
    bc = b * (b == c) # PICK(B,C)

    a = ab | ac | bc # Bitwise OR, safe b/c non-matches are zeroed
    
    result = a + (a == 0) * d - 1 # a or d - 1

    # if upgraded:
    #   return downgrade_type(result)
    # only need to reset data if we weren't upgraded 
    # b/c no copy was made in that case
    data -= 1
    return result

def down_sample_target_count(target, scale=2):
    
    # pdb.set_trace()
    _target = target.data.cpu().numpy()
    _target = _zero_corrected_countless(_target)
    if scale == 4:
        _target = _zero_corrected_countless(_target)
    elif scale == 8:
        _target = _zero_corrected_countless(_target)
        _target = _zero_corrected_countless(_target)
    # pdb.set_trace()
    return torch.from_numpy(_target)

