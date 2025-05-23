import json
from pycocotools.coco import COCO
from tqdm import tqdm

# Initialize the COCO API for instance annotations
dataDir = 'annotations_trainval2017'  # Update this to your COCO dataset path
dataType = 'train2017'  # Change this if you're using a different split (train2017, etc.)
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# Initialize the COCO API for caption annotations
captionAnnFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
coco_caps = COCO(captionAnnFile)

# Categories we're interested in
life_categories = ["person"]

# Get category IDs for our target categories
target_cat_ids = []
for category in life_categories:
    catIds = coco.getCatIds(catNms=[category])
    target_cat_ids.extend(catIds)

# Area threshold (e.g., object must occupy at least 1% of the image)
area_ratio_threshold = 0.2

print(f"Finding images with exactly one object from specified categories and enough area...")
filtered_images = []

# Get all image IDs that contain any of our target categories
for category in tqdm(life_categories):
    catIds = coco.getCatIds(catNms=[category])
    imgIds = coco.getImgIds(catIds=catIds)

    for img_id in imgIds:
        obj_ann_ids = coco.getAnnIds(imgIds=img_id)
        obj_anns = coco.loadAnns(obj_ann_ids)

        target_objects = []
        for ann in obj_anns:
            if ann['category_id'] in target_cat_ids:
                target_objects.append(ann)

        if len(target_objects) == 1:
            target_ann = target_objects[0]
            img_info = coco.loadImgs(img_id)[0]
            img_area = img_info['width'] * img_info['height']
            obj_area = target_ann.get('area', 0)

            if obj_area / img_area >= area_ratio_threshold:
                cat_info = coco.loadCats(target_ann['category_id'])[0]
                filtered_images.append((img_id, cat_info['name']))

print(f"Found {len(filtered_images)} images with exactly one large-enough object from the specified categories")

dataset = []
print("Creating dataset entries for each filtered image...")
for img_id, category in tqdm(filtered_images):
    try:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco_caps.getAnnIds(imgIds=img_id)
        captions = coco_caps.loadAnns(ann_ids)

        item = {
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'coco_url': img_info['coco_url'],
            'width': img_info['width'],
            'height': img_info['height'],
            'captions': [ann['caption'] for ann in captions],
            'category': category
        }

        dataset.append(item)
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")

# Save to JSON
output_file = 'coco_single_life_object_filtered_by_area.json'
with open(output_file, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset created with {len(dataset)} items and saved to {output_file}")

# Summary statistics
category_counts = {}
for _, category in filtered_images:
    category_counts[category] = category_counts.get(category, 0) + 1

print("\nCategory distribution in filtered dataset:")
for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{category}: {count} images")
