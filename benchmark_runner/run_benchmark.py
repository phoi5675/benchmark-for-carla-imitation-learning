#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

from h5_reader import BenchmarH5Reader
from imit_agent import ImitAgent
from carla_agent import CarlaAgent
from ref_colped_agent import ColPedAgent
from result_saver import ResultSaver
import glob
import os
import sys
from tqdm import tqdm
from multiprocessing import Process, Value
from spawnnpc import SpawnNpc
import signal

try:
    import pygame

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
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
    import traceback

    traceback.print_exc()

# ==============================================================================
# -- import controller ---------------------------------------------------------
# ==============================================================================

from game_imitation import *


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- utils ======---------------------------------------------------------------
# ==============================================================================
import tensorflow_yolov3.carla.utils as utils
from tensorflow_yolov3.carla.config import cfg


def draw_bboxes(pygame, display, window_size, bboxes, show_label=True):
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    font_size = 10
    font = pygame.font.Font('freesansbold.ttf', font_size)

    bb_surface = pygame.Surface(window_size)
    bb_surface.set_colorkey((0, 0, 0))

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = (255, 0, 0)

        rect = (coor[0] + 272, coor[1], coor[2] - coor[0], coor[3] - coor[1])
        pygame.draw.rect(display, bbox_color, rect, 3)
        #            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        if show_label and score > 0.55:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            text = font.render(bbox_mess, True, bbox_color)
            display.blit(text, (rect[0], rect[1] - 15))
    display.blit(bb_surface, (0, 0))


def check_timeout(start_time, now, timeout):
    if now - start_time >= timeout:
        return True
    else:
        return False


def is_stuck(col_history):
    if len(col_history) < 4:
        return False
    if col_history[-1].other_actor.id != col_history[-2].other_actor.id and \
            col_history[-1].other_actor.id == col_history[-3].other_actor.id and \
            col_history[-2].other_actor.id == col_history[-4].other_actor.id:
        return True

    for i in range(-2, -5, -1):
        if col_history[i].other_actor.id != col_history[-1].other_actor.id:
            return False
    return True


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    agent = None
    fps = 25
    timeout = 1
    timeout_spd = 10 / 3.6  # km/h to m/sec
    now = 0
    spawn_npc = None
    npc_process = None

    # shared var for spawn_npc
    shared_stop_requested = Value('d', 0.0)

    weather_preset = find_weather_presets()
    train_weather_preset = [preset for preset in weather_preset
                            if preset[1] == 'Clear Noon' or
                            preset[1] == 'Clear Sunset' or
                            preset[1] == 'Mid Rainy Noon' or
                            preset[1] == 'Mid Rain Sunset']
    test_weather_preset = [preset for preset in weather_preset
                           if preset[1] == 'Cloudy Noon' or
                           preset[1] == 'Soft Rain Sunset']
    weather_preset_list = train_weather_preset if args.train_weather else test_weather_preset

    result_saver = ResultSaver()
    result_saver.weather = "train_weather" if args.train_weather else "test_weather"
    result_saver.town = args.map
    result_saver.agent_name = args.agent

    for weather in weather_preset_list:
        result_saver.weather_tlight_vio[weather[1]] = [0, 0, 0]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        client.load_world(args.map)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        server_world = client.get_world()
        settings = server_world.get_settings()
        settings.deterministic_ragdolls = True
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 1 / fps

        server_world.apply_settings(settings)

        hud = HUD(args.width, args.height)
        world = World(server_world, hud, args)
        controller = KeyboardControl(world, False)

        if args.agent == "imitagent":
            agent = ImitAgent(world.player)
        elif args.agent == "carlaagent":
            agent = CarlaAgent(world.player)
        elif args.agent == "colped":
            agent = ColPedAgent(world.player, target_speed=30)
        elif args.agent == "colcar":
            agent = ColPedAgent(world.player, target_speed=30)

        world.agent = agent
        world.radar_sensor.agent = agent
        world.front_camera.agent = agent
        world.collision_sensor.agent = agent

        clock = pygame.time.Clock()

        # set yellow light time of traffic light
        actor_list = server_world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        for traffic_light in lights_list:
            traffic_light.set_yellow_time(0.7)

        # get routes from h5 file
        h5_reader = BenchmarH5Reader()
        all_route_list = BenchmarH5Reader.read_h5_file(int(args.map[4:6]))

        # dummy waypoint -> 처음 실행 시 wp 제대로 만들지 못하는 버그
        route = random.choice(all_route_list)
        start_transform, end_transform = h5_reader.get_start_end_transform_from_h5_list(route)
        agent.set_destination(start_transform, end_transform)

        spawn_npc = SpawnNpc()
        npc_process = Process(target=spawn_npc.spawn_npc,
                              args=(shared_stop_requested, args.number_of_vehicles, args.number_of_walkers))
        npc_process.start()
        timeout = route[h5_reader.h5_dict.index(h5_reader.route_len)] / timeout_spd

        with tqdm(total=args.test_num * len(weather_preset_list), position=0, leave=True) as pbar:
            for weather in weather_preset_list:
                server_world.set_weather(weather[0])
                route_list = BenchmarH5Reader.extract_test_list(all_route_list, args.test_num)
                start_time = hud.simulation_time

                while len(route_list) > 0:
                    # tick_busy_loop(FPS) : 수직동기화랑 비슷한 tick() 함수
                    clock.tick_busy_loop(fps)

                    if controller.parse_events(client, world, clock):
                        raise Exception
                    
                    # 코스 벗어난 경우 -> hlc reset
                    if agent.is_off_course():
                        tqdm.write("off course. reset navigation")
                        agent.set_destination(world.player.get_transform(), end_transform, set_transform=False)

                    world.tick(clock)

                    if settings.synchronous_mode:
                        server_world.tick()  # 서버 시간 tick

                    # server_world.wait_for_tick()  # to move npc vehicle

                    world.render(display)
                    if agent.bounding_boxes is not None:
                        draw_bboxes(pygame, display, (args.width, args.height), agent.bounding_boxes)
                    pygame.display.flip()
                    control = agent.run_step()

                    agent.timestamp = world.hud.simulation_time

                    control.manual_gear_shift = False
                    world.player.apply_control(control)

                    # get distance traveled
                    if hud.simulation_time - now >= 1:
                        now = hud.simulation_time
                        result_saver.distance_traveled += agent.speed / 3.6

                    # check traffic light violation
                    if agent.is_violate_red_junction():
                        # check imitagent traffic light violation
                        if args.agent == "imitagent" and agent.bounding_boxes is not None and \
                                len(agent.bounding_boxes) < 1:
                            tqdm.write("traffic light violated with no bounding boxes detected")
                            result_saver.traffic_light_no_bboxes += 1
                            result_saver.weather_tlight_vio[weather[1]][1] += 1
                        else:
                            tqdm.write("traffic light violated")
                        result_saver.traffic_light_violation += 1
                        result_saver.weather_tlight_vio[weather[1]][0] += 1

                    # count number of red traffic light when vehicle is in junction
                    agent.add_red_light_history()

                    ####################
                    # 한 테스트 종료 조건 #
                    ####################
                    if agent.is_reached_goal() or check_timeout(start_time, hud.simulation_time, timeout) or \
                            is_stuck(agent.collision_history):
                        if check_timeout(start_time, hud.simulation_time, timeout):
                            tqdm.write("timeout")
                            result_saver.timeout += 1
                        elif agent.is_reached_goal():
                            tqdm.write("reached in time")
                            result_saver.success += 1
                        else:
                            tqdm.write("stuck")
                            result_saver.stuck += 1
                        # kill npcs
                        shared_stop_requested.value = 1.0
                        npc_process.join()

                        # save results
                        result_saver.total_run += 1
                        result_saver.calc_collision(agent)
                        agent.collision_history.clear()

                        # red light count
                        result_saver.red_light_count += len(agent.red_light_history)
                        # 0 : traffic light violation, 1 : no bbox, 2 : total violation
                        result_saver.weather_tlight_vio[weather[1]][2] += len(agent.red_light_history)

                        agent.red_light_history.clear()

                        tqdm.write("violated red light : %d, total red light passed : %d" %
                                   (result_saver.traffic_light_violation, result_saver.red_light_count))

                        # start new run
                        control = carla.VehicleControl()
                        control.steer = 0
                        control.throttle = 0
                        control.brake = 0

                        control.hand_brake = 1
                        world.player.apply_control(control)

                        # sleep for a sec to prevent initial weird maneuvering
                        time.sleep(4)

                        route = route_list.pop()
                        start_transform, end_transform = h5_reader.get_start_end_transform_from_h5_list(route)
                        agent.set_destination(start_transform, end_transform)

                        timeout = route[h5_reader.h5_dict.index(h5_reader.route_len)] / timeout_spd
                        start_time = hud.simulation_time

                        # respawn npcs
                        npc_process = Process(target=spawn_npc.spawn_npc,
                                              args=(shared_stop_requested, args.number_of_vehicles,
                                                    args.number_of_walkers))
                        npc_process.start()

                        pbar.update(1)
    except Exception as e:
        logging.error(str(e))

    finally:
        result_saver.is_dynamic_env(args.number_of_vehicles, args.number_of_walkers)

        result_saver.make_res()
        result_saver.show_res()
        result_saver.save_res()

        # kill npcs
        shared_stop_requested.value = 1.0
        npc_process.join()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


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
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--path',
        default='output/',
        help='path for saving data')
    argparser.add_argument(
        '-m', '--map',
        default='Town03',
        help='load a new map')
    argparser.add_argument(
        '--agent',
        default='imitagent',
        choices=['imitagent', 'carlaagent', 'colped', 'colcar'],
        help='choose agent for benchmark test')
    argparser.add_argument(
        '--train_weather',
        action='store_true',
        help='whether to use train weather')
    argparser.add_argument(
        '--test_num',
        default=10,
        type=int,
        help='# of test for each weather. total will be (test_num * 4) if train weather,(test_num * 2 if test weather')
    argparser.add_argument(
        '-n', '--number_of_vehicles',
        metavar='N',
        default=20,
        type=int,
        help='number of vehicles (default: 20)')
    argparser.add_argument(
        '-w', '--number_of_walkers',
        metavar='W',
        default=30,
        type=int,
        help='number of walkers (default: 30)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


def sigint_handler(signum, frame):
    print("SIGINT detected")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    main()
