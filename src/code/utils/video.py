import subprocess
import re
import json


def ffprobe_video_info(filename: str) -> str:
    process = subprocess.Popen(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                               "-show_streams", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    if stderr:
        raise TypeError
    elif not re.search("duration", stdout.decode()):
        raise TypeError
    return stdout.decode()


def get_fps_from_file(filepath: str) -> float:
    with open(filepath) as jsonf:
        js = json.load(jsonf)
    fps = 30
    for st in js["streams"]:
        if st["codec_type"] == "video":
            fps = float(st["nb_frames"]) / float(st["duration"])
    return fps
