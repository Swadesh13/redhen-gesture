import argparse
import os
import subprocess
import json
import numpy as np
from datetime import datetime
from time import time
from utils.video import ffprobe_video_info, get_fps_from_file
from utils.paths import generate_video_paths
from pose.openpose_base import run_openpose
from pose.keypoints import get_keypoints
from data.gestures_data import arrange_detect_data, arrange_train_data, generate_npy_data_detect, generate_npy_data_train, get_hand_movement_times
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

    if args.input_files:
        assert len(args.input_files) == len(args.elan_csv_files), \
            "Number of video files != Number of elan csv files"

if args.window_size:
    WINDOW_SIZE = args.window_size[0]
if args.max_persons:
    MAX_PERSONS = args.max_persons[0]
if args.diff_ratio:
    assert args.diff_ratio[0] < 1
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
all_keypoints_paths = []
file_count = 0
for ind, file_ in enumerate(input_dir_list):
    if args.input_dir:
        file_path = os.path.join(input_dir, file_)
    elif args.input_files:
        file_path = file_
        file_ = os.path.basename(file_)
    try:    # Check if file is a video file by running ffprobe on it and get the info
        video_info = ffprobe_video_info(file_path)
    except:
        continue
    file_count += 1
    output_video_filename, output_video_dir, output_video_path, video_info_path, output_json_dir, keypoints_path = generate_video_paths(
        file_, output_dir, WINDOW_SIZE, MAX_PERSONS, MAX_CHANGE_RATIO)
    output_video_dirs_list.append(output_video_dir)
    with open(video_info_path, "w") as vidf:
        vidf.write(video_info)
    dt = datetime.fromtimestamp(int(time()))
    print(f"[{dt}] Video File name: {output_video_filename}")
    video_info = json.loads(video_info)
    # Run OpenPose on the input video file
    if not os.listdir(output_json_dir):
        if not run_openpose(file_, file_path, output_video_path, output_json_dir, OPENPOSE_BIN):
            continue

    if not os.path.exists(keypoints_path):
        get_keypoints(output_json_dir, keypoints_path, video_info,
                      WINDOW_SIZE, MAX_PERSONS, MAX_CHANGE_RATIO)
    all_keypoints_paths.append(keypoints_path)

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

# get data in numpy files
train_data_paths = []
detect_data_paths = []
if action == "train":
    npy_files_path = os.path.join(output_dir, "npy_files")
    os.makedirs(npy_files_path, exist_ok=True)
    for elan_file, video_dir, keypoints_path in list(zip(elan_csv_files_list, output_video_dirs_list, all_keypoints_paths)):
        if elan_file:
            output_video_filename = os.path.basename(video_dir)
            video_info_file = os.path.join(video_dir,
                                           f"{output_video_filename}_info.json")
            npy_file = os.path.join(video_dir,
                                    f"{output_video_filename}_npy_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
            npy_file_ = os.path.join(npy_files_path,
                                     f"{output_video_filename}_npy_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
            if os.path.exists(npy_file) and os.path.exists(npy_file_):
                train_data_paths.append(npy_file)
                print("All files available")
            else:
                with open(keypoints_path) as jf:
                    keypoints = dict(json.load(jf))
                csv = get_hand_movement_times(elan_file)
                fps = get_fps_from_file(video_info_file)
                data = arrange_train_data(keypoints, csv, fps, MAX_PERSONS)
                # Generates trainable data in numpy
                npy = generate_npy_data_train(data, WINDOW_SIZE)
                with open(npy_file, "wb") as npyf:
                    np.save(npy_file, npy)
                    np.save(npy_file_, npy)
                train_data_paths.append(npy_file)
                dt = datetime.fromtimestamp(int(time()))
                print(f"[{dt}] Created train npy file at {npy_file}")
    print(f"Completed data generation for {len(train_data_paths)} files")

elif action == "detect":
    npy_files_path = os.path.join(output_dir, "npy_files")
    os.makedirs(npy_files_path, exist_ok=True)
    for video_dir, keypoints_path in list(zip(output_video_dirs_list, all_keypoints_paths)):
        output_video_filename = os.path.basename(video_dir)
        video_info_file = os.path.join(video_dir,
                                       f"{output_video_filename}_info.json")
        npy_file = os.path.join(video_dir,
                                f"{output_video_filename}_npy_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
        npy_file_ = os.path.join(npy_files_path,
                                 f"{output_video_filename}_npy_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
        if os.path.exists(npy_file) and os.path.exists(npy_file_):
            detect_data_paths.append(npy_file)
            print("All files available")
        else:
            with open(keypoints_path) as jf:
                keypoints = dict(json.load(jf))
            data = arrange_detect_data(keypoints, MAX_PERSONS)
            # Generates trainable data in numpy
            npy = generate_npy_data_detect(data, WINDOW_SIZE)
            with open(npy_file, "wb") as npyf:
                np.save(npy_file, npy)
                np.save(npy_file_, npy)
            detect_data_paths.append(npy_file)
            dt = datetime.fromtimestamp(int(time()))
            print(f"[{dt}] Created train npy file at {npy_file}")
    print(f"Completed data generation for {len(detect_data_paths)} files")
