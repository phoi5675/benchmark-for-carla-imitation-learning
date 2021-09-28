$train_num=25
$test_num=50
$no_dynamic=0
$map="Town02"
$ped_num=150
$car_num=50
##################################
# parameter for each town
# (map)  : (# of car) (# of ped)
# Town01 : 25, 150
# Town02 : 20, 75
# Town03 : 50, 150 (temp)
# Town05 : 50, 150 (temp)
##################################

# not dynamic
# imitagent
python run_benchmark.py --test_num $test_num -m $map --agent imitagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic
python run_benchmark.py --test_num $train_num --train_weather -m $map --agent imitagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic
# carlaagnet
# python run_benchmark.py --test_num $test_num -m $map --agent carlaagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic
# python run_benchmark.py --test_num $train_num --train_weather -m $map --agent carlaagent --number_of_vehicles $no_dynamic --number_of_walkers $no_dynamic
#
# dynamic
# imitagent
python run_benchmark.py --test_num $test_num -m $map --agent imitagent --number_of_vehicles $car_num --number_of_walkers $ped_num
python run_benchmark.py --test_num $train_num --train_weather -m $map --agent imitagent --number_of_vehicles $car_num --number_of_walkers $ped_num
# carlaagent
# python run_benchmark.py --test_num $test_num -m $map --agent carlaagent --number_of_vehicles $car_num --number_of_walkers $ped_num
# python run_benchmark.py --test_num $train_num --train_weather -m $map --agent carlaagent --number_of_vehicles $car_num --number_of_walkers $ped_num