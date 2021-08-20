import os
from typing import List
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import subprocess


def load_model(filepath: str):
    return tf.keras.models.load_model(filepath)


def detect_on_np_data(model, data):
    assert model.input_shape[2:] == np.array(data).shape[2:], \
        "model input shape and data shape do not match!"
    return model.predict(np.array(data, dtype=np.float32), batch_size=1, verbose=1)


def smoothen_results(results: List, df: pd.DataFrame):
    MAX_CONT_FALSE = 7
    MIN_FRAMES_FOR_GESTURE = 3
    fc = 0
    tr = False
    nfc = 0
    start = 0
    end = 0
    preds = list(np.zeros((len(df)), dtype="int"))
    for frame_r in results:
        if frame_r[1]:
            fc += 1
            tr = True
            nfc = 0
            end = frame_r[0] + 1
            if not start:
                start = frame_r[0] - 2
        else:
            if tr:
                if nfc < MAX_CONT_FALSE:
                    nfc += 1
                else:
                    if fc >= MIN_FRAMES_FOR_GESTURE:
                        tr_li = list(df.loc[(df.frame >= start) &
                                            (df.frame <= end)].index)
                        for j in range(tr_li[0], tr_li[-1]+1):
                            preds[j] = 1
                    start = 0
                    end = 0
                    tr = False
                    fc = 0
                    nfc = 0
    df["gesture"] = list(map(bool, preds))
    return df


def compile_results(frames, preds, df_filepath, threshold=0.5):
    df_data = []
    for [frame, d], result in zip(frames, preds):
        df_data.append([frame,  result[0] > threshold])

    df = pd.DataFrame(df_data, columns=["frame", "gesture"])
    df = smoothen_results(df_data, df)
    df.to_csv(df_filepath, index=False)
    return df


def visualize_prediction(input_video_path: str, output_video_path: str, preds_df: pd.DataFrame):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (frame_width, frame_height))
    ind = 0
    count = 0
    df_ = preds_df.loc[preds_df.gesture == True].reset_index()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out = frame
        if count < len(df_):
            if ind == df_.loc[count, "frame"] and df_.loc[count, "gesture"] == True:
                count += 1
                label = "Hand Gesture"
                x1 = int(frame_width/2 - 10)
                y1 = 50
                scale = 1
                t_size = cv2.getTextSize(
                    label, 0, fontScale=scale, thickness=1)[0]
                out = cv2.rectangle(out, (x1, y1), (x1+t_size[0], y1-t_size[1]-5),
                                    (128, 128, 0), -1, cv2.LINE_AA)
                out = cv2.putText(out, label, (x1, y1-5), 0,
                                  scale, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(out)
        ind += 1

    writer.release()
    cap.release()


def add_audio(v_path: str, orig_va_path: str):
    audio_path = os.path.splitext(orig_va_path)[0]+"_audio.mp3"
    ext = os.path.splitext(v_path)[1]
    audio_extract_args = (
        "ffmpeg", "-y", "-i", f"{orig_va_path}", "-vn", "-acodec", "libmp3lame", f"{audio_path}")
    popen1 = subprocess.Popen(audio_extract_args, stderr=subprocess.PIPE)
    popen1.wait()
    err = popen1.stderr.read()
    if err:
        print(err.decode())
    v_path_ = os.path.splitext(v_path)[0]+"_"+ext
    os.rename(f"{v_path}", f"{v_path_}")
    audio_merge_args = ("ffmpeg", "-y", "-fflags", "+genpts", "-i",
                        f"{v_path_}", "-i", f"{audio_path}", "-c:v", "copy", "-c:a", "aac", f"{v_path}")
    popen2 = subprocess.Popen(audio_merge_args, stderr=subprocess.PIPE)
    popen2.wait()
    err = popen2.stderr.read()
    if err:
        print(err.decode())
    os.remove(v_path_)


def run_detection(model_path: str, data: np.array, frames: np.array, df_filepath: str, input_video_path: str, output_video_path: str, threshold: float = 0.5):
    model = load_model(model_path)
    results = detect_on_np_data(model, data)
    preds = compile_results(frames, results, df_filepath, threshold)
    visualize_prediction(input_video_path, output_video_path, preds)
    add_audio(output_video_path, input_video_path)
