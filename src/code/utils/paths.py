import os
from typing import Tuple


def generate_video_paths(file_: str, output_dir: str, WINDOW_SIZE: int, MAX_PERSONS: int, MAX_CHANGE_RATIO: float) -> Tuple[str, str, str, str, str, str]:
    output_video_filename = os.path.splitext(file_)[0]
    output_video_dir = os.path.join(output_dir, output_video_filename)
    output_video_path = os.path.join(output_video_dir,
                                     f"{output_video_filename}.mp4")
    video_info_path = os.path.join(output_video_dir,
                                   f"{output_video_filename}_info.json")
    output_json_dir = os.path.join(output_video_dir,
                                   f"{output_video_filename}_json")
    keypoints_path = os.path.join(output_video_dir,
                                  f"{output_video_filename}_keypoints_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.json")
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    return output_video_filename, output_video_dir, output_video_path, video_info_path, output_json_dir, keypoints_path
