import argparse
import os
import subprocess
import json
import sys
import numpy as np
from math import ceil
from datetime import datetime
from time import time
from config import MAX_CHANGE_RATIO, MAX_PERSONS, WINDOW_SIZE, MODEL_PATH, TRAIN_VAL_SPLIT
from utils.video import ffprobe_video_info, get_fps_from_file
from utils.paths import generate_video_paths
from pose.openpose_base import run_openpose
from pose.keypoints import get_keypoints
from data.gestures_data import arrange_detect_data, arrange_train_data, generate_npy_data_detect, generate_npy_data_train, get_hand_movement_times
from model.detect import run_detection
from model.train import train

# Command-Line Arguments
parser = argparse.ArgumentParser()

subparser = parser.add_subparsers(dest="action",
                                  help="Whether to train or detect")
subparser.required = True

# openpose_help_parser = subparser.add_parser("openpose_help")
# openpose_help_parser.add_argument("--options", type=str, nargs='+',
#                                   help='additional options to openpsoe help')

train_parser = subparser.add_parser("train")
train_parser.add_argument("--retrain", action='store_true', default=False,
                          help="Retrain from a given model or default model.")
train_parser.add_argument("--batch_size", type=int, nargs=1, default=[16],
                          help="Path to save model after training.")
train_parser.add_argument("--epochs", type=int, nargs=1, default=[20],
                          help="Number of epochs for training.")
train_parser.add_argument("--lr", type=float, nargs=1, default=[2e-4],
                          help="Learning Rate")

detect_parser = subparser.add_parser("detect")

group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument("--input_dir", type=str, nargs=1,
                    help="input directory for one or more readable video files")
group1.add_argument('--input_files', type=str, nargs='+',
                    help="complete path to input files (one or more)")

parser.add_argument('--elan_csv_files', type=str, nargs='+',
                    help="complete path to elan_csv files (one or more) in same order as input video files")
parser.add_argument("--output_dir", type=str, nargs=1, required=True,
                    help="writable output directory for storing output/intermediate files")
parser.add_argument("--window_size", type=int, nargs=1, default=[WINDOW_SIZE],
                    help="Window Size for generating time-series data")
parser.add_argument("--max_persons", type=int, nargs=1, default=[MAX_PERSONS],
                    help="max persons acceptable in a frame")
parser.add_argument("--diff_ratio", type=float, nargs=1, default=[MAX_CHANGE_RATIO],
                    help="Ratio times (height+width)/2 that will be maximum difference between consecutive frames for the same person")
parser.add_argument("--detection_threshold", type=float, nargs=1, default=[0.5],
                    help="max persons acceptable in a frame")
parser.add_argument("--model_path", type=str, nargs=1,
                    help="Path to model for training / detection.")

# Argument parsing and validation
if len(sys.argv) > 1:
    args = parser.parse_args()
else:
    parser.error("No argument provided!")

action = args.action

if not (action == "openpose_help" or ((args.input_dir or args.input_files) and args.output_dir)):
    parser.error(
        "Atleast one of input and output files/directories or openpose_help must be mentioned as argument.")

if args.input_dir and args.input_files:
    parser.error("input_dir and input_files cannot be used together.")

if action == "train":
    if args.input_files:
        if args.elan_csv_files:
            assert len(args.input_files) == len(args.elan_csv_files), \
                "Number of video files != Number of elan csv files"
        else:
            parser.error("input_files requires elan_csv_files while training.")


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
    for elan_file, video_dir, keypoints_path in zip(elan_csv_files_list, output_video_dirs_list, all_keypoints_paths):
        if elan_file:
            output_video_filename = os.path.basename(video_dir)
            video_info_file = os.path.join(video_dir,
                                           f"{output_video_filename}_info.json")
            npy_file = os.path.join(video_dir,
                                    f"{output_video_filename}_npy-train_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
            npy_file_ = os.path.join(npy_files_path,
                                     f"{output_video_filename}_npy-train_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
            if os.path.exists(npy_file) and os.path.exists(npy_file_):
                train_data_paths.append(npy_file)
                print(f"{npy_file} available")
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

    split = ceil(TRAIN_VAL_SPLIT*len(train_data_paths))
    train_files = train_data_paths[:split]
    val_files = None
    if split < len(train_data_paths):
        val_files = train_data_paths[split:]
    x_train, y_train, x_val, y_val = [], [], [], []
    if args.model_path:
        MODEL_PATH = args.model_path[0]

    for video_dir, train_data_path in zip(output_video_dirs_list, train_files):
        output_video_filename = os.path.basename(video_dir)
        with open(train_data_path, "rb") as npf:
            data = np.load(npf, allow_pickle=True)
        for frame, d, lb in data:
            # 1 channel required
            x_train.append(np.array([d], dtype=np.float32))
            y_train.append(lb)
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)

    if val_files:
        for video_dir, val_data_path in zip(output_video_dirs_list, val_files):
            output_video_filename = os.path.basename(video_dir)
            with open(val_data_path, "rb") as npf:
                data = np.load(npf, allow_pickle=True)
            for frame, d, lb in data:
                # 1 channel required
                x_val.append(np.array([d], dtype=np.float32))
                y_val.append(lb)
        x_val = np.array(x_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int8)

    if args.retrain:
        MODEL_PATH = args.model_path[0] if args.model_path else MODEL_PATH
    else:
        MODEL_PATH = None

    train(MODEL_PATH, x_train, y_train,
          x_val, y_val, args.lr[0], args.batch_size[0], args.epochs[0], output_dir)

    print(f"Completed training for {len(train_data_paths)} files")


elif action == "detect":
    npy_files_path = os.path.join(output_dir, "npy_files")
    os.makedirs(npy_files_path, exist_ok=True)
    for video_dir, keypoints_path in zip(output_video_dirs_list, all_keypoints_paths):
        output_video_filename = os.path.basename(video_dir)
        video_info_file = os.path.join(video_dir,
                                       f"{output_video_filename}_info.json")
        npy_file = os.path.join(video_dir,
                                f"{output_video_filename}_npy-detect_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
        npy_file_ = os.path.join(npy_files_path,
                                 f"{output_video_filename}_npy-detect_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.npy")
        if os.path.exists(npy_file) and os.path.exists(npy_file_):
            detect_data_paths.append(npy_file)
            print(f"{npy_file} available")
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

    for video_dir, detect_data_path in zip(output_video_dirs_list, detect_data_paths):
        output_video_filename = os.path.basename(video_dir)
        openpose_video_path = os.path.join(video_dir,
                                           f"{output_video_filename}.mp4")
        output_video_path = os.path.join(video_dir,
                                         f"{output_video_filename}_output.mp4")
        pred_df_path = os.path.join(video_dir,
                                    f"{output_video_filename}_preds-df_w{WINDOW_SIZE}_p{MAX_PERSONS}_r{MAX_CHANGE_RATIO}.csv")
        if args.model_path:
            MODEL_PATH = args.model_path[0]
        with open(detect_data_path, "rb") as npf:
            data = np.load(npf, allow_pickle=True)
        x = []
        for frame, d in data:
            x.append(np.array([d], dtype=np.float32))   # 1 channel required
        run_detection(MODEL_PATH, x, data, pred_df_path, openpose_video_path,
                      output_video_path, args.detection_threshold)

    print(f"Completed detection for {len(detect_data_paths)} files")
