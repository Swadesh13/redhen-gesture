# RedHen Gestures

This is the repository for all the code that was ceated during the GSoC 2021 at Red Hen Lab for the project topic "Red Hen Open Dataset for gestures with performance baselines at Red Hen Lab".

For more info on the work and some information / How to use / documentations on Singularity and HPC, check my blog @ [swadesh13.github.io/gestures-dataset-blog](https://swadesh13.github.io/gestures-dataset-blog)

The OpenPose container used in this project is derived from `frankierr/openpose_containers:focal_nvcaffe` at Docker. (Check the `singularity.df` file for reference). `workflow.py` file works as the entry to all the code and handles all the tasks.

The `src` directory contains all the code. There is a `code` directory for python files such as `workflow.py`. Also, the singularity def file and `.openpose_env` are under the `singularity` directory. I updated the `.openpose_env` and the def files, so I have kept them here.

### Steps followed after screating a singularity sandbox from openpose.def file
* Changed access permissions of `/.openpose_env` to allow execution.
* Changed the `/.openpose_env` to add the workflow.py file path.
* Openpose searches for `models` directory at `OPENPOSE_SRC` i.e. base OpenPose folder. So, copied the directory `/home/opt/openpose_models` to `/home/opt/openpose/models`
* Added this github repo to `/home/opt/openpose/`.

## Structure of the Code

Go to `src/code`. Workflow.py handles all input and outputs. Inside data, data related tasks such as arranging keypoints to numpy data is present in `gestures_data.py`. The model directory contains the training and detection code. The `pose` folder contains code to generate keypoints from OpenPose output in the `keypoints.py` file. `utils` contains some simple utility stuff. `config.py` includes the basic default configuration. `singularity`. `singularity` folder contains the basic def file on which the container is built and also the `.openpose_env` file, which contains an extra environment path to be added. `models` contains the model with default parameters.