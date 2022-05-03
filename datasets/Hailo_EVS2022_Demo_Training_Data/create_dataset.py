import json
from collections import OrderedDict

IMAGE_COUNT = 1050
CLASS_COUNT = 3
ANNOTATION_FILE = "M1401_gt_whole.txt"

def main():
    instances_train = OrderedDict()
    instances_val = OrderedDict()

    # Instances Training
    instances_train['info'] = {
        "year": 2022,
        "version": "1.0",
        "description": "For object detection",
        "date_created": "2022"
    }

    image_list = []
    for i in range(IMAGE_COUNT):
        image_list.append({
            "date_captured": "2022",
            "file_name": "img{:06d}.jpg".format(i+1),
            "id": i+1,
            "height": 540,
            "width": 1024
        })

    instances_train['images'] = image_list
    instances_train['type'] = 'instances'

    annotations = []
    with open(ANNOTATION_FILE,'r') as annotation_file:
        for i, line in enumerate(annotation_file, 1):
            frame_index, _, left, top, width, height, _, _, category = line.split(',')
            annotations.append({
                "bbox": [int(left), int(top), int(width), int(height)],
                "category_id": int(category),
                "id": i,
                "image_id": frame_index,
                "iscrowd": 0,
                "area": int(width)*int(height)
            })
    instances_train['annotations'] = annotations
    
    categories = []
    for i in range(CLASS_COUNT):
        categories.append({
            "id": i+1,
            "name": "{}".format(i),
            "supercategory": "{}".format(i)
        })
    instances_train['categories'] = categories
    
    # Instances Validation
    instances_val['annotations'] = instances_train['annotations'][:5]
    instances_val['categories'] = instances_train['categories']
    instances_val['images'] = instances_train['images'][:5]
    instances_val['info'] = instances_train['info']
    instances_val['type'] = instances_train['type']

    with open('instances_train2017.json', 'w') as json_file:
        json.dump(instances_train, json_file)
    with open('instances_val2017.json', 'w') as json_file:
        json.dump(instances_val, json_file)

if __name__=='__main__':
    main()
