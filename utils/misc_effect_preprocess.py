import cv2
import numpy as np

ROTATE_DEGREE = [90, 180, 270]

def rotate(image, boxes, prob=0.5, border_value=(128, 128, 128)):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    rotate_degree = ROTATE_DEGREE[np.random.randint(0, 3)]
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=border_value)

    new_boxes = []

    for box in boxes:
        x1 = (box[0] - box[2] / 2.) * w
        y1 = (box[1] - box[3] / 2.) * h
        x2 = (box[0] + box[2] / 2.) * w
        y2 = (box[1] + box[3] / 2.) * h
        points = M.dot([
            [x1, x2, x2, x1],
            [y1, y2, y2, y1],
            [1, 1, 1, 1],
        ])
        # Extract the min and max corners again.
        min_xy = np.sort(points, axis=1)[:, :2]
        min_x = np.mean(min_xy[0])
        min_y = np.mean(min_xy[1])
        max_xy = np.sort(points, axis=1)[:, 2:]
        max_x = np.mean(max_xy[0])
        max_y = np.mean(max_xy[1])

        temp_x = ((min_x + max_x) / 2) / new_w
        temp_y = ((min_y + max_y) / 2) / new_h
        temp_w = (max_x - min_x) / new_w
        temp_h = (max_y - min_y) / new_h

        new_boxes.append([temp_x, temp_y, temp_w, temp_h])
    boxes = np.array(new_boxes)
    return image, boxes

def flipx(image, boxes, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    image = image[:, ::-1]
    boxes[:, 0] = 1 - boxes[:, 0]
    return image, boxes

class MiscEffect:
    def __init__(self,rotate_prob=0.05, flip_prob=0.5, crop_prob=0, translate_prob=0,
                 border_value=(128, 128, 128)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.translate_prob = translate_prob
        self.border_value = border_value

    def __call__(self, image, boxes):
        image, boxes = rotate(image, boxes, prob=self.rotate_prob, border_value=self.border_value)
        image, boxes = flipx(image, boxes, prob=self.flip_prob)
        return image, boxes
