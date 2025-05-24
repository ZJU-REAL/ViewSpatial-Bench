import os
import re

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import json
from scannet_utils import *
# Set environment variable to resolve OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def get_align_matrix(meta_file):
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix


def get_3d_bbox_corners(centers, sizes):
    """
    Batch generate 8 corner points for multiple 3D bounding boxes

    Parameters:
       centers: numpy array with shape (N, 3), representing N bounding box centers
       sizes: numpy array with shape (N, 3), representing N bounding box dimensions [length, width, height]

    Returns:
       corners: numpy array with shape (N, 8, 3), representing 8 corner points for N bounding boxes
    """
    N = centers.shape[0]  # 边界框数量
    corners = np.zeros((N, 8, 3))

    for i in range(N):
        x, y, z = centers[i]
        l, w, h = sizes[i] / 2.0

        # 定义8个角点的相对坐标
        corners[i] = np.array([
            [x + l, y + w, z + h], [x + l, y + w, z - h], [x + l, y - w, z + h], [x + l, y - w, z - h],
            [x - l, y + w, z + h], [x - l, y + w, z - h], [x - l, y - w, z + h], [x - l, y - w, z - h]
        ])

    return corners


def draw_3d_bboxes(image, bboxes_2d, visibilities, colors=None, thickness=2, show_invisible=False):
    """
    Batch draw multiple projected 3D bounding boxes on an image

    Parameters:
        image: PIL Image object
        bboxes_2d: numpy array with shape (N, 8, 2), representing N sets of projected 2D points
        visibilities: numpy array with shape (N, 8), indicating point visibility
        colors: list of length N containing N color tuples, auto-generated if None
        thickness: integer representing line thickness
        show_invisible: boolean indicating whether to show invisible edges (displayed as dashed lines)

    Returns:
        image: PIL Image object with bounding boxes drawn
    """
    N = bboxes_2d.shape[0]  # Number of bounding boxes

    # Connection line indices for bounding box edges
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Generate different colors automatically if none provided
    if colors is None:
        colors = []
        for i in range(N):
            # Generate random colors while avoiding colors that are too dark or too bright
            color = (
                np.random.randint(50, 200),
                np.random.randint(50, 200),
                np.random.randint(50, 200)
            )
            colors.append(color)

    # Convert image to OpenCV format
    img_cv = np.array(image)

    # Draw edges for each bounding box
    for i in range(N):
        bbox_2d = bboxes_2d[i]
        visibility = visibilities[i]
        color = colors[i]

        # Draw edges
        for [j, k] in lines:
            pt1 = (int(bbox_2d[j, 0]), int(bbox_2d[j, 1]))
            pt2 = (int(bbox_2d[k, 0]), int(bbox_2d[k, 1]))

            # Draw solid line if both endpoints are visible
            if visibility[j] and visibility[k]:
                cv2.line(img_cv, pt1, pt2, color, thickness)
            # Draw dashed line if show_invisible is set and at least one endpoint is visible
            elif show_invisible and (visibility[j] or visibility[k]):
                # Create dashed line
                pts = np.array([pt1, pt2], np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_cv, [pts], False, color, thickness=1, lineType=cv2.LINE_AA, shift=0)

        # Draw visible points
        for j, vis in enumerate(visibility):
            if vis:
                pt = (int(bbox_2d[j, 0]), int(bbox_2d[j, 1]))
                cv2.circle(img_cv, pt, 3, color, -1)

    # Convert back to PIL Image
    return Image.fromarray(img_cv)

def project_3d_bbox_to_2d(bboxes_3d, intrinsic, pose, image_size, depth_image=None, depth_scale=1000.0,
                          occlusion_threshold=0.1):
    """
    Batch project multiple 3D bounding boxes to 2D image plane and detect occlusion,
    resolving size mismatch between depth map and color image

    Parameters:
        bboxes_3d: numpy array with shape (N, 8, 3), representing 8 corner points of N 3D bounding boxes
        intrinsic: numpy array with shape (4, 4), camera intrinsic matrix
        pose: numpy array with shape (4, 4), camera extrinsic matrix (camera pose)
        image_size: tuple (width, height), representing image dimensions
        depth_image: numpy array with shape (height, width), representing depth image, no occlusion detection if None
        depth_scale: float, scale factor for depth image to convert depth values to meters
        occlusion_threshold: float, depth difference threshold in meters for determining point occlusion

    Returns:
        bboxes_2d: numpy array with shape (N, 8, 2), representing projected 2D points
        visibilities: numpy array with shape (N, 8), indicating point visibility
    """
    N = bboxes_3d.shape[0]  # Number of bounding boxes

    # Initialize results
    bboxes_2d = np.zeros((N, 8, 2))
    visibilities = np.zeros((N, 8), dtype=bool)

    # Get depth image dimensions (if available)
    depth_height, depth_width = 0, 0
    color_width, color_height = image_size
    depth_to_color_scale_x, depth_to_color_scale_y = 1.0, 1.0

    if depth_image is not None:
        depth_height, depth_width = depth_image.shape[:2]
        # Calculate scaling ratio from depth image to color image
        depth_to_color_scale_x = color_width / depth_width
        depth_to_color_scale_y = color_height / depth_height

    # Calculate transformation from world coordinate system to camera coordinate system
    world_to_cam = np.linalg.inv(pose)

    # Process all N objects in the scene
    for i in range(N):
        # Get 8 corner points of current bounding box
        bbox_3d = bboxes_3d[i]

        # Convert 3D bounding box to homogeneous coordinates
        bbox_3d_homogeneous = np.hstack([bbox_3d, np.ones((bbox_3d.shape[0], 1))])  # (8, 4)

        # Transform 3D points from world coordinate system to camera coordinate system
        cam_points = bbox_3d_homogeneous @ world_to_cam.T  # (8, 4)

        # Check if points are in front of camera (z > 0)
        visibility = cam_points[:, 2] > 0

        # Apply projection matrix to project points onto image plane
        points_2d_homogeneous = cam_points @ intrinsic.T  # (8, 4)

        # Perspective division: convert homogeneous coordinates to image coordinates
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]

        # Check if points are within image bounds
        in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < color_width) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < color_height)

        # Update visibility: points must be in front of camera and within image bounds
        visibility = visibility & in_image

        # Detect occlusion if depth image is available
        if depth_image is not None:
            for j in range(8):
                if visibility[j]:
                    # Get pixel coordinates of projected point in color image
                    color_x, color_y = int(points_2d[j, 0]), int(points_2d[j, 1])

                    # Convert color image coordinates to depth image coordinates
                    depth_x = int(color_x / depth_to_color_scale_x)
                    depth_y = int(color_y / depth_to_color_scale_y)

                    # Ensure point is within depth image bounds
                    if 0 <= depth_x < depth_width and 0 <= depth_y < depth_height:
                        # Get actual depth from depth map
                        actual_depth = float(depth_image[depth_y, depth_x]) / depth_scale  # Convert to meters

                        # Get calculated depth (z value in camera coordinate system)
                        calculated_depth = float(cam_points[j, 2])

                        # Compare actual depth with calculated depth to determine occlusion
                        # Point is considered occluded only when depth value is valid (>0) and calculated depth is significantly greater than actual depth
                        if actual_depth > 0 and calculated_depth - actual_depth > occlusion_threshold:
                            visibility[j] = False
                    else:
                        # Maintain current visibility state if point is outside depth image bounds
                        pass

        # Save results for storage and later inclusion in visibility data JSON file
        bboxes_2d[i] = points_2d
        visibilities[i] = visibility

    return bboxes_2d, visibilities


def load_3d_boxes(json_file):
    """
    Load 3D bounding box data from JSON file

    Parameters:
       json_file: file path to JSON file containing 3D bounding box data

    Returns:
       centers: numpy array with shape (N, 3), representing bounding box center points
       sizes: numpy array with shape (N, 3), representing bounding box dimensions
       labels: list containing bounding box labels
       object_ids: list containing bounding box object IDs
    """
    data = json_file

    centers = []
    sizes = []
    labels = []
    object_ids = []

    # 解析JSON数据
    for box in data['boxes']:
        center = np.array(box['center'])
        size = np.array(box['size'])
        label = box.get('label', 'unknown')
        object_id = box.get('object_id', -1)

        centers.append(center)
        sizes.append(size)
        labels.append(label)
        object_ids.append(object_id)

    return np.array(centers), np.array(sizes), labels, object_ids


def process_image_with_boxes(image_path, boxes_json, intrinsic_path, pose_path, meta_file, output_path=None,
                             visibility_json_path=None, depth_image_path=None, depth_scale=1000.0,
                             occlusion_threshold=0.1, draw_picture=False):
    """
    Process a single image by drawing all 3D bounding boxes and saving visible object information,
    with occlusion detection and handling of size mismatch between depth map and color image

    Parameters:
       image_path: file path to the image
       boxes_json: JSON object containing 3D bounding box data
       intrinsic_path: file path to camera intrinsic parameters
       pose_path: file path to camera pose
       meta_file: file path to scene metadata
       output_path: output image path, uses default path if None
       visibility_json_path: visibility JSON file path, uses default path if None
       depth_image_path: depth image file path, no occlusion detection if None
       depth_scale: float, scale factor for depth image to convert depth values to meters
       occlusion_threshold: float, depth difference threshold in meters for determining point occlusion

    Returns:
       output_path: path to the output image
       visibility_json_path: path to the visibility JSON file
    """

    image = Image.open(image_path)
    image_size = image.size

    def cv_imread(file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    # 加载深度图像（如果提供）
    depth_image = None
    if depth_image_path and os.path.exists(depth_image_path):
        depth_image = cv_imread(depth_image_path)
        if len(depth_image.shape) > 2:
            depth_image = depth_image[:, :, 0]

    intrinsic = load_matrix_from_txt(intrinsic_path)
    pose = load_matrix_from_txt(pose_path)

    axis_align_matrix = get_align_matrix(meta_file)

    pose = axis_align_matrix @ pose

    world_to_cam = np.linalg.inv(pose)

    centers, sizes, labels, object_ids = load_3d_boxes(boxes_json)

    bboxes_3d = get_3d_bbox_corners(centers, sizes)

    bboxes_2d, visibilities = project_3d_bbox_to_2d(
        bboxes_3d, intrinsic, pose, image_size, depth_image, depth_scale, occlusion_threshold
    )

    # ------------------------------------------------Store the image file for drawing------------------------------------------------
    if draw_picture:
        unique_labels = list(set(labels))
        label_colors = {}
        for i, label in enumerate(unique_labels):
            h = (i * 30) % 180
            hsv = np.array([[[h, 255, 255]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
            label_colors[label] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        colors = [label_colors[label] for label in labels]
        result_image = draw_3d_bboxes(image, bboxes_2d, visibilities, colors=colors, show_invisible=True)

        img_draw = ImageDraw.Draw(result_image)
        for i, (bbox_2d, visibility, label) in enumerate(zip(bboxes_2d, visibilities, labels)):
            visible_points = bbox_2d[visibility]
            if len(visible_points) > 0:
                top_point = visible_points[np.argmin(visible_points[:, 1])]
                x, y = int(top_point[0]), int(top_point[1] - 10)
                img_draw.text((x, y), label, fill=colors[i])

        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + "_with_boxes.jpg"

        result_image.save(output_path)

    # ------------------------------------------------Store visibility data in JSON format------------------------------------------------
    visibility_data = {
        "image_path": os.path.basename(image_path),
        "visible_objects": []
    }

# It was originally set up, but the obstruction filtering will reduce the visibility by 8 points. Therefore, after being stored in the visibility JSON file,
# you can decide for yourself at what level to filter (I currently set it to be greater than 0.2, because two points indicate that one side of the box can be seen).
    visibility_threshold = 0.01

    bboxes_3d_cam = []
    for bbox_3d in bboxes_3d:
        bbox_3d_homogeneous = np.hstack([bbox_3d, np.ones((bbox_3d.shape[0], 1))])  # (8, 4)
        cam_points = bbox_3d_homogeneous @ world_to_cam.T  # (8, 4)
        bbox_3d_cam = cam_points[:, :3]
        bboxes_3d_cam.append(bbox_3d_cam)

    for i, (bbox_2d, bbox_3d_cam, visibility, label, object_id) in enumerate(
            zip(bboxes_2d, bboxes_3d_cam, visibilities, labels, object_ids)):

        visibility_ratio = np.mean(visibility)
        is_visible = visibility_ratio >= visibility_threshold

        if is_visible:
            visible_points_count = np.sum(visibility)

            bbox_2d_list = bbox_2d.tolist()
            bbox_3d_cam_list = bbox_3d_cam.tolist()

            # Store coordinate center points in their respective coordinate systems
            bbox_2d_cam_center = np.mean(bbox_2d, axis=0)
            bbox_3d_cam_center = np.mean(bbox_3d_cam, axis=0)

            visibility_data["visible_objects"].append({
                "object_id": object_id,
                "label": label,
                "visibility_ratio": float(visibility_ratio),
                "visible_points_count": int(visible_points_count),
                "bbox_2d_center": bbox_2d_cam_center.tolist(),
                "bbox_3d_center": bbox_3d_cam_center.tolist(),
                "vertices_visibility": visibility.tolist(),
                "occlusion_checked": depth_image is not None
            })

    if visibility_json_path is None:
        visibility_json_path = os.path.splitext(output_path)[0] + "_visibility.json"

    with open(visibility_json_path, 'w') as f:
        json.dump(visibility_data, f, indent=2)

    return output_path, visibility_json_path


def batch_process_images(image_folder, image_chosen, boxes_json, intrinsic_path, meta_file, output_folder,
                         visibility_folder, depth_folder, pose_folder, depth_scale=1000.0, occlusion_threshold=0.1, draw_picture=False):
    """
    Batch process all images in a folder with occlusion detection

    Parameters:
       image_folder: folder path containing images
       boxes_json: JSON object containing 3D bounding box data
       intrinsic_path: file path to camera intrinsic parameters
       meta_file: file path to scene metadata
       output_folder: output folder path, uses default path if None
       visibility_folder: visibility JSON folder path, uses output_folder if None
       depth_folder: depth image folder path, no occlusion detection if None
       depth_scale: float, scale factor for depth image to convert depth values to meters
       occlusion_threshold: float, depth difference threshold in meters for determining point occlusion

    Returns:
       processed_images: list of processed image file paths
       visibility_jsons: list of visibility JSON file paths
    """

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(visibility_folder, exist_ok=True)

    # Obtain all image files
    image_files = image_chosen


    processed_images = []
    visibility_jsons = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # Keep the file name as it is, and then change it to "txt" to retrieve the corresponding camera parameters from the "pose" folder.
        pose_file = image_file.replace('.jpg', '.txt')
        pose_path = os.path.join(pose_folder, pose_file)

        if not os.path.exists(pose_path):
            print(f"The pose file does not exist.{pose_path}!")
            continue

        depth_image_path = None
        if depth_folder:
            depth_file = image_file.replace('.jpg', '.png')
            depth_image_path = os.path.join(depth_folder, depth_file)
            if not os.path.exists(depth_image_path):
                print(f"Depth image does not exist: {depth_image_path}")
                depth_image_path = None

        output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_with_boxes.jpg")
        visibility_json_path = os.path.join(visibility_folder, f"{os.path.splitext(image_file)[0]}_visibility.json")

        processed_path, vis_json_path = process_image_with_boxes(
            image_path, boxes_json, intrinsic_path, pose_path, meta_file,
            output_path, visibility_json_path, depth_image_path, depth_scale, occlusion_threshold, draw_picture
        )

        processed_images.append(processed_path)
        visibility_jsons.append(vis_json_path)


    summary_data = {
        "scene": os.path.basename(os.path.dirname(image_folder)),
        "image_count": len(processed_images),
        "depth_images_used": depth_folder is not None,
        "occlusion_threshold": occlusion_threshold,
        "per_image_visibility": []
    }

    # 汇总每个图像的可见性信息
    for vis_json_path in visibility_jsons:
        try:
            with open(vis_json_path, 'r') as f:
                vis_data = json.load(f)
                summary_data["per_image_visibility"].append({
                    "image_path": vis_data["image_path"],
                    "visible_object_count": len(vis_data["visible_objects"]),

                    "visible_object_ids": [obj["object_id"] for obj in vis_data["visible_objects"] if obj["visibility_ratio"] > 0.1],
                    "visible_object_labels": [obj["label"] for obj in vis_data["visible_objects"] if obj["visibility_ratio"] > 0.1],

                    "occlusion_checked": any(obj.get("occlusion_checked", False) for obj in vis_data["visible_objects"])
                })
        except Exception as e:
            print(f"读取可见性文件 {vis_json_path} 时出错: {e}")

    summary_path = os.path.join(visibility_folder, "visibility_summary.json")

    def extract_number(image_path):
        match = re.search(r'(\d+)\.jpg', image_path)
        if match:
            return int(match.group(1))
        return 0

    summary_data['per_image_visibility'] = sorted(summary_data['per_image_visibility'],key=lambda x: extract_number(x["image_path"]))

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    return processed_images, visibility_jsons


def get_3d_box(scene_name, pointcloud_folder, label_map_file):
    scan_path = f"{pointcloud_folder}/{scene_name}"

    scan_name = os.path.split(scan_path)[-1]
    mesh_file = os.path.join(scan_path, scan_name + "_vh_clean_2.ply")
    agg_file = os.path.join(scan_path, scan_name + ".aggregation.json")
    seg_file = os.path.join(scan_path, scan_name + "_vh_clean_2.0.010000.segs.json")
    meta_file = os.path.join(
        scan_path, scan_name + ".txt"
    )  # includes axisAlignment info for the train set scans.
    mesh_vertices, label_ids, instance_ids, instance_bboxes, object_id_to_label_id, json_boxes = export(
        mesh_file, agg_file, seg_file, meta_file, label_map_file
    )
    return json_boxes


def process(scene_name, draw_picture=False):
   # Original dataset path (modifiable)
   scan_path = f"/datasets/scannet/data/scans/{scene_name}"

   # Get all RGB-D images in the scene (modifiable)
   image_folder = f"/datasets/scannet/scenes/{scene_name}/mc_frames"

   # Get all point cloud files and label mapping file in the scene
   pointcloud_folder = "/datasets/scannet/scans"
   label_map_file = "/datasets/scannet/scannetv2-labels.combined.tsv"

   # Output folders (modifiable)
   output_folder = f"scannet_metadata/{scene_name}/output_images"  # Store rendered images
   visibility_folder = f"scannet_metadata/{scene_name}/visibility_data"  # Store object information, coordinates, and visibility data for each image

   image_all_files = os.listdir(image_folder)
   image_chosen = [file for file in image_all_files if file.lower().endswith('.jpg')]

   # Depth image file path
   depth_folder = os.path.join(scan_path, "depth")
   # Camera parameter file path
   pose_folder = os.path.join(scan_path, "pose")
   # Camera intrinsic file path, only RGB-D intrinsic data is used here
   intrinsic_path = os.path.join(scan_path, "intrinsic_color.txt")
   # For obtaining alignment matrix
   meta_file = os.path.join(
       pointcloud_folder, scene_name, scene_name + ".txt"
   )

   boxes_json = get_3d_box(scene_name, pointcloud_folder, label_map_file)

   # Set depth image scale factor and occlusion threshold
   depth_scale = 1000.0  # Assume depth units are in millimeters, convert to meters
   occlusion_threshold = 0.1  # Set depth difference threshold to 10 centimeters

   # Batch process all images
   processed_images, visibility_jsons = batch_process_images(
       image_folder, image_chosen, boxes_json, intrinsic_path, meta_file,
       output_folder, visibility_folder, depth_folder, pose_folder, depth_scale, occlusion_threshold, draw_picture
   )



if __name__ == "__main__":

    # draw_picture boolean value determines whether to draw new images to the output folder (drawing every 100 scenes as shown below)
    # Open file and read each line
    with open('scannetv2_val.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    scenes = [line.strip() for line in lines]

    for i, scene in enumerate(scenes):
        draw_picture = False
        if i%100 == 0:
            draw_picture = True
            print(f"Processed {i} scenes")
        scene_name = scene
        process(scene_name, draw_picture)