import tensorflow as tf
import numpy as np
import pandas as pd
import cv2


def load_model(filepath: str):
    return tf.keras.models.load_model(filepath)


def detect_on_np_data(model, data):
    assert model.input_shape[2:] == np.array(data).shape[2:], \
        "model input shape and data shape do not match!"
    return model.predict(np.array(data, dtype=np.float32), batch_size=1, verbose=1)


def compile_results(frames, preds, df_filepath, threshold=0.5):
    df_data = []
    for [frame, d], result in zip(frames, preds):
        df_data.append([frame,  result[0] > threshold])

    df = pd.DataFrame(df_data, columns=["frame", "gesture"])
    df.to_csv(df_filepath, index=False)

    return df


def visualize_prediction(input_video_path, output_video_path, preds_df):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'PIM1'),
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
                out = cv2.rectangle(
                    out, (x1, y1), (x1+t_size[0], y1-t_size[1]-5), (128, 128, 0), -1, cv2.LINE_AA, )
                out = cv2.putText(out, label, (x1, y1-5), 0,
                                  scale, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(out)
        ind += 1

    writer.release()
    cap.release()


def run_detection(model_path, data, frames, df_filepath, input_video_path, output_video_path, threshold=0.5):
    model = load_model(model_path)
    results = detect_on_np_data(model, data)
    preds = compile_results(frames, results, df_filepath, threshold)
    visualize_prediction(input_video_path, output_video_path, preds)
