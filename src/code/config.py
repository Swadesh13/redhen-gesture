MAX_PERSONS = 6
WINDOW_SIZE = 5
MAX_CHANGE_RATIO = 0.025
REQUIRED_KEYPOINTS = ["neck", "right shoulder", "right elbow", "right wrist",
                      "left shoulder", "left elbow", "left wrist", ]
TRAIN_COLS = ["neck_x", "neck_y", "right shoulder_x", "right shoulder_y", "right elbow_x", "right elbow_y", "right wrist_x",
              "right wrist_y", "left shoulder_x", "left shoulder_y", "left elbow_x", "left elbow_y", "left wrist_x", "left wrist_y"]
MODEL_PATH = "/opt/openpose/redhen-gesture/src/model.h5"
