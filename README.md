# RedHen Gestures

This is the repository for all the code that was ceated during the GSoC 2021 at Red Hen Lab for the project topic "Red Hen Open Dataset for gestures with performance baselines at Red Hen Lab".

The OpenPose container used in this project is derived from `frankierr/openpose_containers:focal_nvcaffe` at Docker. (Check the `singularity.df` file for reference). `workflow.py` file works as the entry to all the code and handles all the tasks.

The `src` directory contains all the code. There is a `code` directory for python files such as `workflow.py`. Also, the singularity def file and `.openpose_env` are under the `singularity` directory. I updated the `.openpose_env` and the def files, so I have kept them here.