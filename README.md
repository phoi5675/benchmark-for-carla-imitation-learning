# benchmark-for-carla-imitation-learning
carla benchmark for imitation learning running on CARLA 0.9.11(NOT official!)

an autonomous driving benchmark running on CARLA simulator.

benchmark content is based on [End-to-end Driving via Conditional Imitation Learning
](https://arxiv.org/abs/1710.02410)

for detailed info like data collecting, training model, etc for our model, please see [carlaIL](https://github.com/phoi5675/carlaIL)

tested on CARLA 0.9.11

# benchmark_maker
make benchmark routes for maps

# benchmark runner
run benchmark based on your autonomous driving model and course

if you want to run a sample benchmark based on our model and carla imitation-learning model, download [trained models](https://drive.google.com/file/d/1OeKvHv0nNpDCXzKiBaeU9RB3EA9y9Bes/view?usp=sharing) and unzip it in benchmark_runner/

# requirements(only necessary ones)
use same or lower version of these packages. higher version might cause error :(
- tensorflow==1.15.X
- hypy==2.9.0

# acknowledgements
- carla for [CARLA Simulator](https://carla.org/)
- project's base autonomous driving model for [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
- traffic light detection model for [YOLOv3 implemented by Tensorflow 1.X](https://github.com/YunYang1994/tensorflow-yolov3)

# paper
this is a project for undergraduate thesis.

[paper(korean)](https://lib.kau.ac.kr/mir.liberty.file/libertyfile/contents/0000000002/20220106035239503NZ9I6HQOIG.pdf)
