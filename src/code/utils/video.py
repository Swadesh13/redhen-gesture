import subprocess
import re


def ffprobe_video_info(filename: str):
    process = subprocess.Popen(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                               "-show_streams", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    if stderr:
        raise TypeError
    elif not re.search("duration", stdout.decode()):
        raise TypeError
    return stdout.decode()
