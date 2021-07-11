import argparse
import os
import subprocess
import json
import numpy as np
from utils.video import ffprobe_video_info, get_fps_from_file
from pose.keypoints import get_all_keypoints, divide_keypoints_fn, divide_keypoints_count, divide_keypoints_position, arrange_persons
from train.gestures_data import arrange_train_data, generate_npy_data, get_hand_movement_times
from config import MAX_CHANGE_RATIO, MAX_PERSONS, WINDOW_SIZE

parser = argparse.ArgumentParser("Arguments for using OpenPose")

parser.add_argument("action", type=str, nargs=1,
                    help="Whether train or detect or get openpose_help", choices=["train", "detect", "openpose_help"])
parser.add_argument("--input_dir", type=str, nargs=1,
                    help="input directory for one or more readable video files")
parser.add_argument('--input_files', type=str, nargs='+',
                    help="complete path to input files (one or more)")
parser.add_argument('--elan_csv_files', type=str, nargs='+',
                    help="complete path to elan_csv files (one or more) in same order as input video files")
parser.add_argument("--output_dir", type=str, nargs=1,
                    help="writable output directory for storing output/intermediate files")
parser.add_argument("--window_size", type=int, nargs=1,
                    help="Window Size for generating time-series data")
parser.add_argument("--max_persons", type=int, nargs=1,
                    help="max persons acceptable in a frame")
parser.add_argument("--diff_ratio", type=float, nargs=1,
                    help="Ratio of (height+width)/2 that will be maximum difference between consecutive frames for the same person")


args = parser.parse_args()

action = args.action[0]

if not (action == "openpose_help" or ((args.input_dir or args.input_files) and args.output_dir)):
    parser.error(
        "Atleast one of input and output files/directories or openpose_help must be mentioned as argument.")

if args.input_dir and args.input_files:
    parser.error("input_dir and input_files cannot be used together.")

if action == "train":
    if (not args.input_files) ^ (not args.elan_csv_files):
        parser.error("input_files requires elan_csv_files while training.")

assert len(args.input_files) == len(args.elan_csv_files), \
    "Number of video files != Number of elan csv files"

if args.window_size:
    WINDOW_SIZE = args.window_size[0]
if args.max_persons:
    MAX_PERSONS = args.max_persons[0]
if args.diff_ratio:
    MAX_CHANGE_RATIO = args.diff_ratio[0]

ENV = dict(os.environ)
OPENPOSE_BIN = ENV["OPENPOSE_BIN"]

if action == "openpose_help":
    openpose_help_args = (f"{OPENPOSE_BIN}", "--help")
    proc = subprocess.Popen(openpose_help_args, stdout=subprocess.PIPE)
    proc.wait()
    print(proc.stdout.read().decode())
    exit()

output_dir = args.output_dir[0]
if args.input_dir:
    input_dir = args.input_dir[0]
    assert os.access(input_dir, os.R_OK), \
        f"{input_dir} : Input Directory has to be readable"
    input_dir_list = os.listdir(input_dir)
elif args.input_files:
    assert os.access(args.input_files[0], os.R_OK), \
        f"{args.input_files[0]} : Input File has to be readable"
    input_dir_list = args.input_files

assert os.access(output_dir, os.W_OK), \
    f"{output_dir} : Output Directory has to be writable"
output_video_dirs_list = []
elan_csv_files_list = []
file_count = 0
for ind, fil in enumerate(input_dir_list):
    if args.input_dir:
        file_path = os.path.join(input_dir, fil)
    elif args.input_files:
        file_path = fil
        fil = os.path.basename(fil)
    try:    # Check if file is a video file by running ffprobe on it
        video_info = ffprobe_video_info(file_path)
    except:
        continue
    file_count += 1
    output_video_filename = os.path.splitext(fil)[0]
    output_video_dir = os.path.join(output_dir, output_video_filename)
    output_video_dirs_list.append(output_video_dir)
    output_video_path = os.path.join(output_video_dir,
                                     f"{output_video_filename}.avi")
    video_info_path = os.path.join(output_video_dir,
                                   f"{output_video_filename}_info.json")
    output_json_dir = os.path.join(output_video_dir,
                                   f"{output_video_filename}_json")
    keypoints_path = os.path.join(output_video_dir,
                                  f"{output_video_filename}_keypoints.json")
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    with open(video_info_path, "w") as vidf:
        vidf.write(video_info)
    video_info = json.loads(video_info)
    if not os.path.exists(output_video_path):
        print(
            f"Running OpenPose on {fil}. Get the output video file at {output_video_path} and json files at {output_json_dir}")
        openpose_args = (f"{OPENPOSE_BIN}", "--video", f"{file_path}", "--display", "0",
                         "--write-video", f"{output_video_path}", "--write-json", f"{output_json_dir}")
        popen = subprocess.Popen(openpose_args, stderr=subprocess.PIPE)
        popen.wait()
        err = popen.stderr.read()
        if err:
            print(err.decode())
            break

        print(f"Done running openpose on {fil}.")

    if not os.path.exists(keypoints_path):
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

    if action == "train":
        if args.input_dir:
            csv_file = os.path.splitext(file_path)[0]+".csv"
            if os.access(csv_file, os.R_OK):
                elan_csv_files_list.append(csv_file)
            else:
                elan_csv_files_list.append('')
                print("Could not access CSV file for gesture training at" +
                      os.path.splitext(file_path)[0]+".csv")
        elif args.input_files:
            if os.access(args.elan_csv_files[ind], os.R_OK):
                elan_csv_files_list.append(args.elan_csv_files[ind])
            else:
                elan_csv_files_list.append('')
                print("Could not access CSV file for gesture training at" +
                      args.elan_csv_files[0])

# train, get csv data
if action == "train":
    train_data_paths = []
    npy_files_path = os.path.join(output_dir, "npy_files")
    os.makedirs(npy_files_path, exist_ok=True)
    for elan_file, video_dir in list(zip(elan_csv_files_list, output_video_dirs_list)):
        if elan_file:
            output_video_filename = os.path.basename(video_dir)
            video_info_file = os.path.join(video_dir,
                                           f"{output_video_filename}_info.json")
            keypoints_path = os.path.join(video_dir,
                                          f"{output_video_filename}_keypoints.json")
            npy_file = os.path.join(video_dir,
                                    f"{output_video_filename}_npy.npy")
            npy_file_ = os.path.join(
                npy_files_path, f"{output_video_filename}_npy.npy")
            hand_gesture_times = get_hand_movement_times(elan_file)
            fps = get_fps_from_file(video_info_file)
            with open(keypoints_path) as jf:
                keypoints = dict(json.load(jf))
            csv = get_hand_movement_times(elan_file)
            fps = get_fps_from_file(video_info_file)
            data = arrange_train_data(keypoints, csv, fps, MAX_PERSONS)
            # Generates trainable data in numpy
            npy = generate_npy_data(data, WINDOW_SIZE)
            with open(npy_file, "wb") as npyf:
                np.save(npy_file, npy)
                np.save(npy_file_, npy)
            train_data_paths.append(npy_file)
