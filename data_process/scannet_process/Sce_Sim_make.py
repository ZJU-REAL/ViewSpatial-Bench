import itertools
import random
import os, sys
import inspect
from collections import Counter
from frame_sampling import get_full_images
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from scannet_utils import *


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = (
                data["segGroups"][i]["objectId"] + 1
            )  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file):
    """points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = read_label_mapping(
        label_map_file, label_from="raw_category", label_to="nyu40id"
    )
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [
                (xmin + xmax) / 2,
                (ymin + ymax) / 2,
                (zmin + zmax) / 2,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin,
                label_id,
            ]
        )
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox


    return (
        mesh_vertices,
        label_ids,
        instance_ids,
        instance_bboxes,
        object_id_to_label_id
    )

def get_3d_box(scene_name, pointcloud_folder, label_map_file):
    scan_path = f"{pointcloud_folder}/{scene_name}"

    scan_name = os.path.split(scan_path)[-1]
    mesh_file = os.path.join(scan_path, scan_name + "_vh_clean_2.ply")
    agg_file = os.path.join(scan_path, scan_name + ".aggregation.json")
    seg_file = os.path.join(scan_path, scan_name + "_vh_clean_2.0.010000.segs.json")
    meta_file = os.path.join(
        scan_path, scan_name + ".txt"
    )  # includes axisAlignment info for the train set scans.
    mesh_vertices, label_ids, instance_ids, instance_bboxes, object_id_to_label_id = export(
        mesh_file, agg_file, seg_file, meta_file, label_map_file
    )
    return instance_bboxes


def calculate_relative_position(A, B, C):
    A, B, C = map(np.array, (A, B, C))

    vector_AB = B - A
    if np.linalg.norm(vector_AB) < 1e-6:
        raise ValueError("Objects A and B are at the same position.")

    forward = vector_AB / np.linalg.norm(vector_AB)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)

    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
    else:
        right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    vector_AC = C - A
    local_x = np.dot(vector_AC, right)
    local_y = np.dot(vector_AC, up)
    local_z = np.dot(vector_AC, forward)

    return local_x, local_y, local_z


def get_direction(local_x, local_z):
    angle = np.degrees(np.arctan2(local_x, local_z))
    angle = (angle + 360) % 360

    if 22.5 <= angle < 67.5:
        return "front-right"
    elif 67.5 <= angle < 112.5:
        return "right"
    elif 112.5 <= angle < 157.5:
        return "back-right"
    elif 157.5 <= angle < 202.5:
        return "back"
    elif 202.5 <= angle < 247.5:
        return "back-left"
    elif 247.5 <= angle < 292.5:
        return "left"
    elif 292.5 <= angle < 337.5:
        return "front-left"
    else:
        return "front"


def generate_qa_pairs(obj1, obj2, obj3, label1, label2, label3):
    """Generate QA pairs describing the relative position."""
    try:
        x, y, z = calculate_relative_position(obj1, obj2, obj3)
    except ValueError:
        return []

    direction = get_direction(x, z)
    if direction == "same position":
        return []

    qa_templates = [
        (f"If you stand at {label1} facing {label2}, where is {label3}?",
         f"If I stand at {label1} and face {label2}, then {label3} would be to my {direction}."),

        (f"Imagine standing at {label1} looking towards {label2}, where is {label3}?",
         f"Picture me standing at {label1}, facing {label2}—then {label3} would be on my {direction}."),

        (f"When positioned at {label1} facing {label2}, where can you find {label3}?",
         f"From my vantage point at {label1}, with my eyes fixed on {label2}, {label3} is located to my {direction}."),

        (f"Standing at {label1}, gazing at {label2}, where should {label3} be?",
         f"From this spot at {label1}, looking directly at {label2}, I’d locate {label3} on my {direction} side.")
    ]
    # All possible options
    all_options = ["left", "right", "front", "back", "back-right", "back-left",
                  "front-left", "front-right"]
    qa_pairs = []

    q_template = [random.choice(qa_templates)]
    for q, a in q_template:
        distractors = [opt for opt in all_options if opt not in direction and direction not in opt]
        selected_distractors = random.sample(distractors, 3)
        options = [direction] + selected_distractors
        random.shuffle(options)
        option_letters = ["A", "B", "C", "D"]
        correct_letter_index = options.index(direction)
        correct_option = f"{option_letters[correct_letter_index]}. {direction}"
        formatted_options = "\n".join([f"{option_letters[i]}. {options[i]}" for i in range(4)])
        question = f"{q}\n{formatted_options}"
        qa_pairs.append({
            "question": question,
            "answer": correct_option
        })

    return qa_pairs


def get_random_combinations(lst, max_samples=10000):
    all_combinations = list(itertools.combinations(lst, 3))
    num_samples = min(max_samples, len(all_combinations))
    return random.sample(all_combinations, num_samples)

def get_jpg_files(folder_path):
    jpg_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
            jpg_files.append(filename)

    return jpg_files


if __name__ == "__main__":
    nyu40_to_category = {
        0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed",
        5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window",
        10: "bookshelf", 11: "picture", 12: "counter", 13: "blinds",
        14: "desk", 15: "shelves", 16: "curtain", 17: "dresser",
        18: "pillow", 19: "mirror", 20: "floor mat", 21: "clothes",
        22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
        26: "paper", 27: "towel", 28: "shower curtain", 29: "box",
        30: "whiteboard", 31: "person", 32: "nightstand", 33: "toilet",
        34: "sink", 35: "lamp", 36: "bathtub", 37: "bag",
        38: "other structure", 39: "other furniture", 40: "other prop"
    }

    scene_root = "scannet_metadata"
    output_path = r"scannet_metadata/perspective_3d.json"

    # Get all point cloud files and label mapping file in the scene
    pointcloud_folder = "/datasets/scannet/scans"
    label_map_file = "/datasets/scannet/scannetv2-labels.combined.tsv"

    qa_dataset = []
    scene_num = 0

    with open('scannetv2_val.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    scenes = [line.strip() for line in lines]
    for i, scene in enumerate(scenes):
        scene_name = scene

        img_size = (1296, 968)
        instance_bboxes = get_3d_box(scene_name, pointcloud_folder, label_map_file)

        # 3D bounding box
        bboxes_3d = [tuple(bbox) for bbox in instance_bboxes if bbox[6] not in [0, 1, 2, 22, 38, 39, 40]]

        bbox_6_counts = Counter(bbox[6] for bbox in bboxes_3d)
        unique_bboxes_3d = [bbox for bbox in bboxes_3d if bbox_6_counts[bbox[6]] == 1]
        le = len(unique_bboxes_3d)
        if le < 3:
            continue
        scene_num = scene_num+1

        combinations_3d = get_random_combinations(unique_bboxes_3d, 40)

        for combination in combinations_3d:
            obj1 = (combination[0][0], combination[0][1],combination[0][2])
            obj2 = (combination[1][0], combination[1][1], combination[1][2])
            obj3 = (combination[2][0], combination[2][1], combination[2][2])

            category_name1 = nyu40_to_category.get(int(combination[0][6]), "unknown")
            category_name2 = nyu40_to_category.get(int(combination[1][6]), "unknown")
            category_name3 = nyu40_to_category.get(int(combination[2][6]), "unknown")
            labels = (category_name1, category_name2, category_name3)

            jpg_files_list = get_full_images(scene_name, labels)
            if not jpg_files_list:
                img_path = os.path.join(scene_root, scene_name)
                img_path = img_path + "/original_images"
                jpg_files_list = get_jpg_files(img_path)
            jpg_files_list = [scene_name + "/original_images/" + a for a in jpg_files_list]
            # generate QA-pairs
            qa_set = generate_qa_pairs(obj1, obj2, obj3, *labels)

            for num, qa in enumerate(qa_set[:4], 1):
                qa_dataset.append({"image_path": scene_name, "question": qa['question'], "answer": qa['answer'], "image": jpg_files_list})

        print(f"Scene {i} has been successfully saved!")

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(qa_dataset, output_file, ensure_ascii=False, indent=4)
    print(f"QA data has been saved to {output_path}, total of {scene_num} scenes! {len(qa_dataset)} questions!")