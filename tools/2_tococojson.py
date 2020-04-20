import json
import os
import cv2
root_path = r'/home/cvos/PythonCode/dataset1'
phase = 'val'
dataset = {'categories': [], 'annotations': [], 'images': []}
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
indexes = os.listdir(os.path.join(root_path, 'images/',phase))
global count
count = 0

with open(os.path.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

    for k, index in enumerate(indexes):
        count += 1
        im = cv2.imread(os.path.join(root_path, 'images/',phase+'/') + index)
        height, width, _ = im.shape

        dataset['images'].append({'file_name': index,
                                  'id': k,
                                  'width': width,
                                  'height': height})
 
        for i, anno in enumerate(annos):
            parts = anno.strip().split()

            if parts[0] == index:

                cls_id = parts[1]
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id),
                    'id': i,
                    'image_id': k,
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
 
        print('{} images handled'.format(count))
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)
