# benchmark maker
a benchmark route maker 

you can make route for default provided maps(Town##)

not tested for custom maps

# how to run
- you must launch CARLA before run benchmark_maker.py
- "carla" folder provided in this repo should be in the same directory with this folder. to use basic_agent.py
```
- (your root folder)
 |- benchmark_maker
 |- carla
   |- agents
     |- navigation
       |- basic_agent.py
```
- if you want to use your own basic_agent.py, then change value self._hop_resolution to 1.0
```
self._hop_resolution = 1.0
```
- create "output" folder if not exists. created routes will be saved in that folder.
- run benchmark_maker.py
```
python benchmark_maker.py
```
- if you want to change map, add argument below. ## should be a town number you want to make route. 
```
-m Town##
```
  you can see list of provided maps using [config.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/util/config.py).
  for more information about files in carla/PythonAPI/, please see carla readthedocs.
```
python config.py --list
```
- the default route filter setting of benchmark maker is route length >= 150m and has at least one turn(left / right) on crossroad.
if you want to change route length, add argument below. ### should be integer and use distance in metric.
```
-l ###
```

- for example, if you want make route for Town05 and route length >= 250,
```
python benchmark_maker.py -m Town05 -l 250
```
