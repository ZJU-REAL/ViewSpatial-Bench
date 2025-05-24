# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Ref: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts """
import os
import sys
import json
import csv

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)


def represents_int(s):
    """if string s represents an int."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_mesh_vertices(filename):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices


def read_mesh_vertices_rgb(filename):
    """read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices



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




def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None, json_file=None):
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
    axis_align_matrix = np.eye(4)
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = np.array([
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]).reshape((4, 4))
            break

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
        label_id = label_map.get(label, 0)
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
    json_boxes = {"boxes": []}

    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id.get(obj_id, 0)
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        if label_id in [0, 1, 2, 22, 38, 39, 40]:
            continue
        # Compute axis-aligned bounding box
        xmin, ymin, zmin = np.min(obj_pc, axis=0)
        xmax, ymax, zmax = np.max(obj_pc, axis=0)
        bbox = np.array([
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            (zmin + zmax) / 2,
            xmax - xmin,
            ymax - ymin,
            zmax - zmin,
            label_id,
        ])
        instance_bboxes[obj_id - 1, :] = bbox

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

        json_boxes["boxes"].append({
            "center": bbox[:3].tolist(),
            "size": bbox[3:6].tolist(),
            "label": nyu40_to_category.get(label_id),
            "object_id": int(obj_id)
        })

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file + "_vert.npy", mesh_vertices)
        np.save(output_file + "_sem_label.npy", label_ids)
        np.save(output_file + "_ins_label.npy", instance_ids)
        np.save(output_file + "_bbox.npy", instance_bboxes)

    return (
        mesh_vertices,
        label_ids,
        instance_ids,
        instance_bboxes,
        object_id_to_label_id,
        json_boxes
    )

