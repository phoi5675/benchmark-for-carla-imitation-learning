#!/bin/sh

train_num=4
test_num=8
no_dynamic=0

# town01, not dynamic
python run_benchmark.py --test_num $test_num -m Town01 --agent imitagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic
python run_benchmark.py --test_num $train_num --train_weather -m Town01 --agent imitagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic

python run_benchmark.py --test_num $test_num -m Town01 --agent carlaagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic
python run_benchmark.py --test_num $train_num --train_weather -m Town01 --agent carlaagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic

# town01, dynamic
python run_benchmark.py --test_num $test_num -m Town01 --agent imitagent
python run_benchmark.py --test_num $train_num --train_weather -m Town01 --agent imitagent

python run_benchmark.py --test_num $test_num -m Town01 --agent carlaagent
python run_benchmark.py --test_num $train_num --train_weather -m Town01 --agent carlaagent