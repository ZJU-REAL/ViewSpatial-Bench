# from paths import *
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from PIL import Image
import json
from utils import *
from inference import get_3angle
from tqdm import tqdm
import os
import requests
import json


def get_keypoint_coordinates(keypoints, index):
    """
    Get keypoint coordinates and visibility from keypoints list at specified index.
    keypoints: list of length 51, containing x, y, v for 17 keypoints.
    index: keypoint index, range [0, 16].
    return: (x, y, v)
    """
    x = keypoints[index * 3]
    y = keypoints[index * 3 + 1]
    v = keypoints[index * 3 + 2]
    return x, y, v


def get_azimuth_direction(azimuth: float) -> str:
    """
    Determine direction name based on azimuth angle

    Args:
       azimuth: azimuth angle in degrees

    Returns:
       direction name (front, front-right, right side, etc.)
    """
    # Normalize angle to 0-360 range
    azimuth = azimuth % 360

    if 337.5 <= azimuth or azimuth < 22.5:
        return "back"
    elif 22.5 <= azimuth < 67.5:
        return "back-left"
    elif 67.5 <= azimuth < 112.5:
        return "left"
    elif 112.5 <= azimuth < 157.5:
        return "front-left"
    elif 157.5 <= azimuth < 202.5:
        return "front"
    elif 202.5 <= azimuth < 247.5:
        return "front-right"
    elif 247.5 <= azimuth < 292.5:
        return "right"
    elif 292.5 <= azimuth < 337.5:
        return "back-right"
    else:
        return "wrong"


annotations_file_path = 'annotations_trainval2017/annotations/person_keypoints_train2017.json'

# Read COCO annotation file
with open(annotations_file_path, 'r') as f:
    coco_data = json.load(f)

def get_ket_and_bbox(image_id):
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    return annotations[0]['keypoints'], annotations[0]['bbox']


def analyze_head_turn(
        image: Image.Image,
        bbox: list,  # [x1, y1, x2, y2]
        keypoints: list,  # [[x, y, conf], ..., [x, y, conf]] length at least 7
        dino,
        val_preprocess,
        device
):
    # Step 1: Crop person image from bbox
    x1, y1, x2, y2 = map(int, bbox)

    # Correct bbox coordinate order to ensure top-left corner comes first
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Check for out-of-bounds coordinates (prevent exceeding image boundaries)
    img_width, img_height = image.size
    x1 = max(0, min(x1, img_width - 1))
    x2 = max(0, min(x2, img_width))
    y1 = max(0, min(y1, img_height - 1))
    y2 = max(0, min(y2, img_height))

    person_image = image.crop((x1, y1, x2, y2))

    # Keypoint indices
    left_shoulder_idx = 5
    right_shoulder_idx = 6

    # Get keypoint coordinates and visibility
    left_shoulder = get_keypoint_coordinates(keypoints, left_shoulder_idx)
    right_shoulder = get_keypoint_coordinates(keypoints, right_shoulder_idx)
    if left_shoulder[2] == 0 or right_shoulder[2] == 0:
        return False, False, False, False, False

    # Step 2: Get left and right shoulder y coordinates (relative to cropped image)
    left_shoulder_y = left_shoulder[1] - y1
    right_shoulder_y = right_shoulder[1] - y1

    cut_y = int(min(left_shoulder_y, right_shoulder_y))

    # Prevent abnormal cut_y values
    if cut_y <= 0 or cut_y >= (y2 - y1):
        return False, False, False, False, False

    # Step 3: Segment head/body images
    head_image = person_image.crop((0, 0, person_image.width, cut_y))
    body_image = person_image.crop((0, cut_y, person_image.width, person_image.height))

    if head_image.height == 0 or head_image.width == 0 or body_image.height == 0 or body_image.width == 0:
        head_image = person_image
        body_image = person_image

    # Step 4: Call model to get angles
    head_angles = get_3angle(head_image, dino, val_preprocess, device)
    body_angles = get_3angle(body_image, dino, val_preprocess, device)

    azimuth_head = float(head_angles[0])
    azimuth_body = float(body_angles[0])

    # Step 5: Determine head turn direction
    def relative_head_direction(az_head, az_body):
        delta = (az_head - az_body + 540) % 360 - 180

        if -90 <= delta < -60:
            return "left"
        elif -60 <= delta < -20:
            return "front-left"
        elif -20 <= delta <= 20:
            return "front"
        elif 20 < delta <= 60:
            return "front-right"
        elif 60 < delta <= 90:
            return "right"
        else:
            return "wrong"

    direction = relative_head_direction(azimuth_head, azimuth_body)

    return azimuth_head, azimuth_body, direction, float(head_angles[3]), float(body_angles[3])


ckpt_path = "dino_weight.pt"

save_path = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = DINOv2_MLP(
    dino_mode='large',
    in_dim=1024,
    out_dim=360 + 180 + 180 + 2,
    evaluate=True,
    mask_dino=False,
    frozen_back=False
)

dino.eval()
print('model create')
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
print('weight loaded')
val_preprocess = AutoImageProcessor.from_pretrained("dinov2-large", cache_dir='./')


def check_image_path(image_path):
    if os.path.exists(image_path):
        return True
    else:
        return False


# ========== Utility Functions ==========
def download_image(img_path, url):
   """Download image and save to specified path"""
   try:
       r = requests.get(url, timeout=10)
       if r.status_code == 200:
           with open(img_path, 'wb') as f:
               f.write(r.content)
           return True
       else:
           print(f"Download failed with status code: {r.status_code}")
           return False
   except Exception as e:
       print(f"Download failed: {e}")
       return False


DATASET_FILE = 'coco_single_life_object_filtered_by_area.json'
with open(DATASET_FILE, 'r') as f:
    dataset = json.load(f)
result = []
for item in tqdm(dataset):
    if item['category'] != 'person':
        continue
    file_name = item['file_name']
    image_path = "train2017/" + file_name
    if not check_image_path(image_path):
        success = download_image(image_path, item['coco_url'])
        if not success:
            print("Download Failed!")
            continue
    origin_image = Image.open(image_path).convert('RGB')
    keypoints, bbox = get_ket_and_bbox(item['image_id'])

    try:
        azimuth_head, azimuth_body, direction, head_confidence, body_confidence = analyze_head_turn(origin_image, bbox,
                                                                                                    keypoints, dino,
                                                                                                    val_preprocess,
                                                                                                    device)
    except:
        continue
    if azimuth_head == False:
        continue

    angles = get_3angle(origin_image, dino, val_preprocess, device)
    azimuth = float(angles[0])
    polar = float(angles[1])
    rotation = float(angles[2])
    confidence = float(angles[3])
    one = {
        'image_id': item['image_id'],
        'coco_url': item['coco_url'],
        'width': item['width'],
        'height': item['height'],
        'captions': item['captions'],
        "azimuth": azimuth,
        "overall_confidence": confidence,
        'azimuth_head': azimuth_head,
        'azimuth_body': azimuth_body,
        'person_direction': direction,
        'camera_direction_v1': get_azimuth_direction(azimuth),
        'camera_direction_v2': get_azimuth_direction(azimuth_head),
        'head_confidence': head_confidence,
        'body_confidence': body_confidence
    }
    result.append(one)

# Save to JSON
output_file = 'train_data.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Dataset created with {len(result)} items and saved to {output_file}")