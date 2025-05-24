import json


def find_continuous_frames(json_data, target_labels):
    """
    Find consecutive video frames that collectively contain the specified three objects

    Parameters:
       json_data (dict): parsed JSON data
       target_labels (list): list of three target object labels

    Returns:
       list: list of consecutive frame filenames
    """
    # Record visible target objects in each frame
    frames_info = []
    for frame in json_data["per_image_visibility"]:
        visible_targets = set(frame["visible_object_labels"]) & set(target_labels)
        frames_info.append({
            "image_path": frame["image_path"],
            "visible_targets": visible_targets,
            "visible_count": len(visible_targets)
        })

    # Find the frame where target objects first appear
    start_index = None
    for i, frame in enumerate(frames_info):
        if frame["visible_count"] > 0:
            start_index = i
            break

    if start_index is None:
        return []

    # Starting from the first appearance frame, search for consecutive frame sequences until all target objects are covered
    current_index = start_index
    found_targets = set()

    while current_index < len(frames_info) and len(found_targets) < len(target_labels):
        found_targets.update(frames_info[current_index]["visible_targets"])
        current_index += 1

        if len(found_targets) == len(target_labels) or current_index == len(frames_info):
            if len(found_targets) == len(target_labels) and current_index != len(frames_info):
                current_index += 1
            break

    # Check if all the target objects have been found
    if len(found_targets) < len(target_labels):
        return []

    result_frames = [frames_info[i]["image_path"] for i in range(start_index, current_index)]
    return result_frames

def get_full_images(scene_name, target_labels):

    with open(f'scannet_metadata/{scene_name}/visibility_data/visibility_summary.json', 'r') as file:
        data = json.load(file)
    result = find_continuous_frames(data, target_labels)
    return result
