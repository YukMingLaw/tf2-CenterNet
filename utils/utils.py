import numpy as np
import tensorflow as tf
import cv2
import math

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def compute_gt(batch_box,input_size,num_classes,max_objects):
    batch_hms = np.zeros((len(batch_box), input_size, input_size, num_classes),dtype=np.float32)
    batch_whs = np.zeros((len(batch_box), max_objects, 2), dtype=np.float32)
    batch_regs = np.zeros((len(batch_box), max_objects, 2), dtype=np.float32)
    batch_reg_masks = np.zeros((len(batch_box), max_objects), dtype=np.float32)
    batch_indices = np.zeros((len(batch_box), max_objects), dtype=np.float32)
    for b,(image,bboxes) in enumerate(zip(batch_box, batch_box)):
        for i in range(len(bboxes)):
            bbox = bboxes[i].copy()
            cls_id = int(bbox[4])
            w = bbox[2] * input_size
            h = bbox[3] * input_size
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([bbox[0] * input_size, bbox[1] * input_size])
                ct_int = ct.astype(np.int32)
                draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius)
                batch_whs[b, i] = 1. * w, 1. * h
                batch_indices[b, i] = ct_int[1] * input_size + ct_int[0]
                batch_regs[b, i] = ct - ct_int
                batch_reg_masks[b, i] = 1
    return [batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices]


