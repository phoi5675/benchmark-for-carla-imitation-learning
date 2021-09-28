#!/usr/bin/env python
"""
need to import BasicAgent in agent.navigation.basic_agent.py
BasicAgent's hop resolution must be 1.0 to get accurate length
"""
import carla
import argparse
import random
import logging


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from benchmark_h5 import BenchmarkH5


def make_vehicle(world, map):
    blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_points = map.get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

    return world.try_spawn_actor(blueprint, spawn_point)


def find_all_routes(args):
    # 서버 연결
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = client.get_world()

    print('load map %r.' % args.map)
    world = client.load_world(args.map)

    map = world.get_map()
    route_agent = BasicAgent(make_vehicle(world, map))
    spawn_points = map.get_spawn_points()

    h5_list = BenchmarkH5()
    h5_list.town_index = int(args.map[4:6])

    with tqdm(total=len(spawn_points) ** 2, position=0, leave=True) as pbar:
        for start_loc in spawn_points:
            for end_loc in spawn_points:
                route_agent.set_destination(start_loc, end_loc)
                # 길이가 짧거나(<=10m) 교차로 회전이 없는 루트는 넣지 않음
                pbar.update(1)
                if len(route_agent.route_trace) <= args.minimum_route_len\
                        or not BenchmarkH5.has_turn(route_agent.route_trace):
                    continue
                if route_agent.is_in_junction():
                    continue

                h5_list.route_to_h5_list(start_loc, end_loc, len(route_agent.route_trace))
                tqdm.write("route length : %d" % len(route_agent.route_trace))

    h5_list.save_h5_file()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        default='Town03',
        help='load a new map')
    argparser.add_argument(
        '-l', '--minimum_route_len',
        default=150,
        type=int,
        help='minimum length for route (default: 150m)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    find_all_routes(args)


if __name__ == '__main__':
    main()
