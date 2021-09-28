#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """
import logging

import tensorflow as tf

import numpy as np
from PIL import Image
import os
import sys
import math
import random
import cv2
import time
from network import Network
import tensorflow_yolov3.carla.utils as utils

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    import traceback
    traceback.print_exc()

import carla
from carla import ColorConverter as cc

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance


class CarlaAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20, image_cut=[115, 510]):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(CarlaAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0 / 20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict})
        self._hop_resolution = 0.5
        self._path_seperation_hop = 3
        self._path_seperation_threshold = 1.0
        self._target_speed = target_speed
        self._grp = None

        self.route_trace = None

        # data from vehicle
        self.speed = 0
        self._radar_data = None
        self._obstacle_ahead = False

        # load network
        g1 = tf.Graph()

        with g1.as_default():
            self.drive_network = Network(model_name='Network', model_dir='/model_carla_agent/')

        self._image_cut = image_cut
        self._image_size = (88, 200, 3)  # 아마 [세로, 가로, 차원(RGB)] 인듯?

        self.front_image = None

        self.bounding_boxes = None

    def set_destination(self, start_loc, end_loc, set_transform=True):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(start_loc.location)
        end_waypoint = self._map.get_waypoint(end_loc.location)

        control_reset = carla.VehicleControl()
        control_reset.steer, control_reset.throttle, control_reset.brake = 0.0, 0.0, 0.0

        start_loc.location.z += 0.5

        if set_transform:
            self._vehicle.apply_control(control_reset)
            self._vehicle.set_transform(start_loc)

        self.route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert self.route_trace

        self._local_planner.set_global_plan(self.route_trace)

        self._local_planner.change_intersection_hcl(enter_hcl_len=15, exit_hcl_len=21)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        self._state = AgentState.NAVIGATING
        # standard local planner behavior
        self._local_planner.buffer_waypoints()

        direction = self.get_high_level_command(convert=False)
        v = self._vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)  # use m/s
        self.speed = speed * 3.6  # use km/s

        control = self._compute_action(self.front_image, speed, direction)
        return control

    def _compute_action(self, rgb_image, speed, direction=None):
        """
        Calculate steer, gas, brake from image input
        :return: carla.VehicleControl
        """
        '''
        # TODO scipy 제대로 되는지 확인
        # scipy 에서 imresize 가 depreciated 됐으므로 다른 방법으로 이미지 리사이즈
        # 이미지를 비율에 어느 정도 맞게 크롭 (395 * 800)
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        # 크롭한 이미지를 리사이징. 비율에 맞게 조절하는게 아니라 조금 찌그러지게 리사이징함. 원래 비율은 352 * 800
        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])
        '''

        rgb_image.convert(cc.Raw)

        array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
        array = array[self._image_cut[0]:self._image_cut[1], :, :3]  # 필요 없는 부분을 잘라내고
        array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        image_pil = image_pil.resize((self._image_size[1], self._image_size[0]))  # 원하는 크기로 리사이즈
        image_input = np.array(image_pil, dtype=np.dtype("uint8"))

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._compute_function(image_input, speed, direction, self.drive_network)

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(acc)
        control.brake = float(brake)

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _compute_function(self, image_input, speed, control_input, network):

        branches = network.network_tensor
        x = network.input_images
        dout = network.dout
        input_speed = network.input_data[1]

        image_input = image_input.reshape(
            (1, network.image_size[0], network.image_size[1], network.image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed / 25.0)

        speed = speed.reshape((1, 1))

        if control_input == RoadOption.LEFT:
            all_net = branches[2]
        elif control_input == RoadOption.RIGHT:
            all_net = branches[3]
        elif control_input == RoadOption.STRAIGHT:
            all_net = branches[1]
        elif control_input == RoadOption.LANEFOLLOW:
            all_net = branches[0]
        else:
            all_net = branches[0]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(network.dropout_vec)}

        output_all = network.sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        predicted_speed = network.sess.run(branches[4], feed_dict=feedDict)
        predicted_speed = predicted_speed[0][0]
        real_speed = speed * 25.0

        real_predicted = predicted_speed * 25.0
        if real_speed < 2.0 and real_predicted > 3.0:
            # If (Car Stooped) and
            #  ( It should not have stopped, use the speed prediction branch for that)

            predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

            predicted_brake = 0.0

            predicted_acc = predicted_acc[0][0]

        return predicted_steers, predicted_acc, predicted_brake

    def set_radar_data(self, radar_data):
        self._radar_data = radar_data
