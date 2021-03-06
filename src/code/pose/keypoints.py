from datetime import datetime
from time import time
from typing import Dict, List, Tuple
import json
import os
from .openpose_base import BODY_25_JOINTS


def get_body_25_keypoints_from_json(filename: str) -> List[Dict]:
    """
    Load keypoints from OpenPose output JSON files. Also, removes confidence scores and keeps only necessary keypoints data.
    """
    with open(filename) as jsonf:
        try:
            keypoints = json.load(jsonf)
        except ValueError:
            return None

    body_25_keypoints = []

    for person in keypoints["people"]:
        pose_keypoints_2d = person["pose_keypoints_2d"]
        # (x, y, confidence)
        keypoints_triplets = [(pose_keypoints_2d[i], pose_keypoints_2d[i+1],
                               pose_keypoints_2d[i+2]) for i in range(0, len(pose_keypoints_2d)-2, 3)]
        keypoints_triplets_body25_map = dict(
            zip(BODY_25_JOINTS, keypoints_triplets))
        body_25_keypoints.append(keypoints_triplets_body25_map)

    return body_25_keypoints


def get_all_keypoints(folder: str, MAX_PERSONS: int) -> Dict:
    """
    Load all frames keypoints. Form a dict.
    """
    json_files = os.listdir(folder)
    all_keypoints = {}
    for jfile in json_files:
        if jfile.endswith("json"):
            abspath = os.path.join(folder, jfile)
            body_25_keypoints = get_body_25_keypoints_from_json(abspath)
            if body_25_keypoints and len(body_25_keypoints) <= MAX_PERSONS:
                # ! Remove those with no. of persons > threshold, say 6
                fn = jfile.split("_")[-2]
                body_25_keypoints_dict = {}
                body_25_keypoints_dict["keypoints"] = body_25_keypoints
                body_25_keypoints_dict["frame_no"] = int(fn)
                body_25_keypoints_dict["count"] = len(body_25_keypoints)
                all_keypoints[fn] = body_25_keypoints_dict
    return all_keypoints


def divide_keypoints_fn(keypoints: Dict, WINDOW_SIZE: int) -> Dict:
    """
    Divide on the basis of missing frames (due to multiple persons with no gesture annotations)
    """
    keys = list(sorted(keypoints.keys()))
    gestures_dict = {}
    gesture_fn = []
    frame_count = 0
    gesture_count = 1
    for i, key in enumerate(keys):
        gesture_fn.append(keypoints[key])
        frame_count += 1
        if i < len(keys)-1:
            if keypoints[key]["frame_no"] != keypoints[keys[i+1]]["frame_no"] - 1:
                if frame_count >= WINDOW_SIZE:
                    gestures_dict[str(gesture_count)] = gesture_fn
                    gesture_count += 1
                    frame_count = 0
                    gesture_fn = []
        else:   # for the last value
            if frame_count >= WINDOW_SIZE:
                gestures_dict[str(gesture_count)] = gesture_fn
                gesture_count += 1
                frame_count = 0
                gesture_fn = []
    return gestures_dict


def divide_keypoints_count(keypoints: Dict, WINDOW_SIZE: int) -> Dict:
    """
    Divide on the basis of count of perons in the frame (change of persons -> change in frame)
    """
    keys = list(keypoints.keys())
    gestures_dict = {}
    gesture_fn = []
    for key in keys:
        gesture_count = 1
        frame_count = 0
        for j, frame in enumerate(keypoints[key]):
            if frame["count"] == 0:
                continue
            gesture_fn.append(frame)
            frame_count += 1
            if j < len(keypoints[key])-1:
                if frame["count"] != keypoints[key][j+1]["count"]:
                    if frame_count >= WINDOW_SIZE:
                        gestures_dict[key+"_"+str(gesture_count)] = gesture_fn
                        gesture_count += 1
                        frame_count = 0
                        gesture_fn = []
            else:   # for the last value
                if frame_count >= WINDOW_SIZE:
                    gestures_dict[key+"_"+str(gesture_count)] = gesture_fn
                    gesture_count += 1
                    frame_count = 0
                    gesture_fn = []
    return gestures_dict


# Divide on the basis of count of relative positions of perons in the frame
# (change of hand / body position more than a margin -> change in shot / different person)
# For now, Relative to 1 person is considered sufficient.


def check_frames_continuous(keypoint1: Dict, keypoint2: List, max_diff: float) -> Tuple[bool, int]:
    neck_keypoint1 = keypoint1["neck"]
    nose_keypoint1 = keypoint1["nose"]
    for i, keypoint in enumerate(keypoint2):
        neck_keypoint2 = keypoint["neck"]
        nose_keypoint2 = keypoint["nose"]
        if (abs(neck_keypoint1[0] - neck_keypoint2[0]) <= max_diff and abs(neck_keypoint1[1] - neck_keypoint2[1]) <= max_diff) or (abs(nose_keypoint1[0] - nose_keypoint2[0]) <= max_diff and abs(nose_keypoint1[1] - nose_keypoint2[1]) <= max_diff):
            return True, i
    return False, i


def divide_keypoints_position(keypoints: Dict, video_info: Dict, WINDOW_SIZE: int, MAX_CHANGE_RATIO: float) -> Dict:
    """
    Check poisiton of keypoints in 2 frames. Threshold determines if continuous frames or shot/angle is changed.
    """
    keys = list(keypoints.keys())
    gestures_dict = {}
    gesture_fn = []
    frame_count = 0
    gesture_count = 1
    vid_width = video_info["width"]
    vid_height = video_info["height"]
    # Maximum allowed change in position
    max_allowed_change = (vid_height+vid_width)/2*MAX_CHANGE_RATIO
    for key in keys:
        for j, frame in enumerate(keypoints[key]):
            gesture_fn.append(frame)
            frame_count += 1
            if j < len(keypoints[key])-1:
                if not check_frames_continuous(frame["keypoints"][0], keypoints[key][j+1]["keypoints"], max_allowed_change)[0]:
                    if frame_count >= WINDOW_SIZE:
                        gestures_dict[key+"_"+str(gesture_count)] = gesture_fn
                        gesture_count += 1
                        frame_count = 0
                        gesture_fn = []
            else:   # for the last value
                if frame_count >= WINDOW_SIZE:
                    gestures_dict[key+"_"+str(gesture_count)] = gesture_fn
                    gesture_count += 1
                    frame_count = 0
                    gesture_fn = []
        gesture_count = 1
    return gestures_dict


def arrange_persons(keypoints: Dict, video_info: Dict, MAX_CHANGE_RATIO: float) -> Dict:
    """
    Arrange the keypoints per person (again compare the neck and nose keypoints)
    """
    keys = list(keypoints.keys())
    gestures_dict = {}
    gesture_fn = []
    vid_width = video_info["width"]
    vid_height = video_info["height"]
    # Maximum allowed change in position
    max_allowed_change = (vid_height+vid_width)/2*MAX_CHANGE_RATIO
    for key in keys:
        start_frame_no = keypoints[key][0]["frame_no"]
        end_frame_no = keypoints[key][-1]["frame_no"]
        if keypoints[key][0]["count"] == 1:
            person_keypoints = []
            for frame in keypoints[key]:
                person_keypoints.append(frame["keypoints"][0])
            gestures_dict[key] = {
                "1": {
                    "person_keypoints": person_keypoints,
                }
            }
            gestures_dict[key].update({
                "start_frame": start_frame_no,
                "end_frame": end_frame_no
            })

        else:
            gestures_dict[key] = {}
            start_frame_no = keypoints[key][0]["frame_no"]
            end_frame_no = keypoints[key][-1]["frame_no"]
            for j, person_keypoint in enumerate(keypoints[key][0]["keypoints"]):
                person_keypoints = []
                person_keypoints.append(person_keypoint)
                for frame in keypoints[key][1:]:
                    _, i = check_frames_continuous(
                        person_keypoints[-1], frame["keypoints"], max_allowed_change)
                    person_keypoints.append(frame["keypoints"][i])
                gestures_dict[key].update({
                    str(j+1): {
                        "person_keypoints": person_keypoints,
                    }
                })
            gestures_dict[key].update({
                "start_frame": start_frame_no,
                "end_frame": end_frame_no
            })

    return gestures_dict


def get_keypoints(output_json_dir: str, keypoints_path: str, video_info: Dict, WINDOW_SIZE: int, MAX_PERSONS: int, MAX_CHANGE_RATIO: float):
    """
    Coordinates all keypoint handling functions
    """
    keypoints = get_all_keypoints(output_json_dir, MAX_PERSONS)
    dkt = divide_keypoints_fn(keypoints, WINDOW_SIZE)
    dkc = divide_keypoints_count(dkt, WINDOW_SIZE)
    for stream in video_info["streams"]:
        if stream["codec_type"] == "video":
            info = stream
            break
    dkp = divide_keypoints_position(
        dkc, info, WINDOW_SIZE, MAX_CHANGE_RATIO)
    ap = arrange_persons(dkp, info, MAX_CHANGE_RATIO)
    with open(keypoints_path, "w") as jf:
        json.dump(ap, jf)
    dt = datetime.fromtimestamp(int(time()))
    print(f"[{dt}] Created keypoints at {keypoints_path}")
