import argparse
import os
import subprocess
import re

def filelength(filePath):
    process = subprocess.Popen(['ffmpeg',  '-i', filePath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(), re.DOTALL).groupdict()
    process.kill()

def mkdir(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass


parser = argparse.ArgumentParser("Arguments for using OpenPose")

parser.add_argument("--input_dir", type=str, nargs=1, help="input directory for one or more readable video files")
parser.add_argument("--output_dir", type=str, nargs=1, help="writable output directory for storing output/intermediate files")
parser.add_argument("-openpose_help", action="store_true", help="Get Help message for OpenPose module")

args = parser.parse_args()

if not (args.openpose_help or (args.input_dir and args.output_dir)):
    parser.error("Atleast one of input and output directories or openpose help must be mentioned as argument.")

ENV = dict(os.environ)
OPENPOSE_BIN = ENV["OPENPOSE_BIN"]

if args.openpose_help:
    openpose_help_args = (f"{OPENPOSE_BIN}", "--help")
    proc = subprocess.Popen(openpose_help_args, stdout=subprocess.PIPE)
    proc.wait()
    print(proc.stdout.read().decode())
    exit()

output_dir = args.output_dir[0]
input_dir = args.input_dir[0]

assert os.access(output_dir, os.W_OK), f"{output_dir} : Output Directory has to be writable"
assert os.access(input_dir, os.R_OK), f"{input_dir} : Input Directory has to be readable"
output_video_dir = os.path.join(output_dir, "openpose_output_videos")
mkdir(output_video_dir)
input_dir_list = os.listdir(input_dir)
file_count = 0
for fil in input_dir_list:
    file_path = os.path.join(input_dir, fil)
    try:    # Check if file is a video file by running ffmpeg on it
        filelength(file_path)
    except:
        continue
    file_count += 1
    output_video_filename = os.path.splitext(fil)[0]
    output_video_path = os.path.join(output_video_dir, f"{output_video_filename}.avi")
    output_json_dir = os.path.join(output_dir, f"{output_video_filename}_json")
    print(f"Running OpenPose on {fil}. Get the output video file at {output_video_path} and json files at {output_json_dir}")
    mkdir(output_json_dir)
    openpose_args = (f"{OPENPOSE_BIN}", "--video", f"{file_path}", "--display", "0", "--write-video", f"{output_video_path}.avi", "--write-json", f"{output_json_dir}")
    popen = subprocess.Popen(openpose_args, stderr=subprocess.PIPE)
    popen.wait()
    err = popen.stderr.read()
    if err:
        print(err.decode())
        exit()

if not file_count:
    os.rmdir(output_video_dir)

print(f"Done running openpose on {file_count} files.")
