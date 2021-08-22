from datetime import datetime
from time import time
import subprocess


BODY_25_JOINTS = [
    "nose",
    "neck",
    "right shoulder",
    "right elbow",
    "right wrist",
    "left shoulder",
    "left elbow",
    "left wrist",
    "mid hip",
    "right hip",
    "right knee",
    "right ankle",
    "left hip",
    "left knee",
    "left ankle",
    "right eye",
    "left eye",
    "right ear",
    "left ear",
    "left big toe",
    "left small toe",
    "left heel",
    "right big toe",
    "right small toe",
    "right heel",
]


def run_openpose(file_: str, file_path: str, output_video_path: str, output_json_dir: str, OPENPOSE_BIN: str, ):
    """
    Runs OpenPose on the video. OpenPose binary present at OPENPOSE_BIN.
    """
    dt = datetime.fromtimestamp(int(time()))
    print(f"[{dt}] Running OpenPose on {file_}. Get the output video file at {output_video_path} and json files at {output_json_dir}")
    openpose_args = (f"{OPENPOSE_BIN}", "--video", f"{file_path}", "--display", "0",
                     "--write-video", f"{output_video_path}", "--write_video_with_audio", "--write-json", f"{output_json_dir}")
    popen = subprocess.Popen(openpose_args, stderr=subprocess.PIPE)
    popen.wait()
    err = popen.stderr.read()
    if err:
        print(err.decode())
        # return False
    print()
    return True
