# benchmark-for-carla-imitation-learning
carla benchmark for imitation learning running on CARLA 0.9.11(NOT official!)


# Overview
an autonomous driving benchmark running on CARLA simulator.

benchmark content is based on [End-to-end Driving via Conditional Imitation Learning
](https://arxiv.org/abs/1710.02410)

tested on CARLA 0.9.11

# benchmark_maker
make benchmark routes for maps

# benchmark runner
run benchmark based on your autonomous driving model and course

# requirements(only necessary ones)
use same or lower version of these packages. higher version might cause error :(
- tensorflow==1.15.X
- hypy==2.9.0

# acknowledgements
- carla for [CARLA Simulator](https://carla.org/)
- project's base autonomous driving model for [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
- traffic light detection model for [YOLOv3 implemented by Tensorflow 1.X](https://github.com/YunYang1994/tensorflow-yolov3)

# paper
this is a project for undergraduate thesis, but the paper hasn't released on my university's library.

instead, I've uploaded on google drive.

[paper(korean)](https://drive.google.com/file/d/1Po2KdzNZ0QiEM0sU_TtCc9wesyc2q1hN/view?usp=sharing)

[presentation video(korean)](https://drive.google.com/file/d/13PeE7181RUUNDKQD5I01NV9hP1l8SX2M/view?usp=sharing)
