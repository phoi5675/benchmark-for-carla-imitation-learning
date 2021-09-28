# Overview
benchmark for autonomous driving models in carla

detailed benchmark conditions are same as original version.

for more informations about benchmark conditions(weathers, number of other cars, peds, etc),

please see [End-to-end Driving via Conditional Imitation Learning
](https://arxiv.org/abs/1710.02410) and [CARLA: An Open Urban Driving Simulator
](https://arxiv.org/abs/1711.03938)

# how to run original model and our pretrained model
- extract routes using [benchmark_maker](https://github.com/phoi5675/benchmark-for-carla-imitation-learning/tree/main/benchmark_maker)
and move route file into benchmark_runner/route_data/
- download [pretrained models] and move folders in zip file into benchmark_runner/
- run CARLA simulator
- run benchmark using example command below
our model
```
python run_benchmark.py --test_num 2 -m Town01 --agent imitagent
```
original model
```
python run_benchmark.py --test_num 2 -m Town01 --agent carlaagent
```
- after the benchmark ends, the result of benchmark will be printed on terminal and saved in result folder in txt format like below.
```
result for agent : imitagent, map : Town01, weather : test_weather, dynamic : True
total	success	timeout	stuck	col_car	col_ped	col_oth	lht_vio	no_bbx	totrdlt	traveled(m)
66	38	24	4	21	1	7	57	40	165	42761.00
traffic light violation details
weather		tlht_vio	no_bbx	totrdlt
Cloudy Noon	38	25	115	
Soft Rain Sunset	19	15	50	
```
- to see more options about run_benchmark.py, run command below
```
python run_benchmark.py --help
```
- shell(linux) or ps1(windows) might be useful if you want to run benchmark multiple times at different conditions.
see benchmark.sh or benchmark.ps1 for more info.

# explanation for files in this folder
- folders
  - result : benchmark results will be saved in here
  - route_data : folder for route data(ex: Town##_all_routes.h5)
- benchmark.ps1 / .sh : automated benchmark runner in multiple different conditions(models, weather, # of peds, cars, ...)
for windows(.ps1) and linux(.sh)
- carla_agent.py, imitation_learning_network.py : code for original model
- __game_imitation.py : sensors are declared in this file__
- h5_reader.py : reads routes in route_data/Town##_all_routes.h5
- imit_agent.py, network.py : code for our model
- ref_colped_agent.py : code for reference model. this model does not stop when other car or pedestrian is ahead
- result_saver.py : counts data and save benchmark result
- __run_benchmark.py : core file for running benchmark__
- spawnnpc.py : spawns other cars & pedestrians in the map. based on [generate_traffic.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py)

# test your own model(NOT TESTED)
__as I've not tested other model, there might be bugs. you might need to find solutions on your own!__
- write your own "agent" model which inherits Agent class from agent.py in this repo.
your "agent" must have
  - set_destination() method; you can copy from our imit_agent.py file
  - get high-level command when run_step() method is called to get routes
- attach sensors to your agent. sensors for your model should be declared in game_imitation.py
- add your agent declaration in [run_benchmark.py](https://github.com/phoi5675/benchmark-for-carla-imitation-learning/blob/main/benchmark_runner/run_benchmark.py#L207)
- run benchmark using method above
