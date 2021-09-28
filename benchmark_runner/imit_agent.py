#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

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


class ImitAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20, image_cut=[115, 510]):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(ImitAgent, self).__init__(vehicle)

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
        g2 = tf.Graph()

        with g1.as_default():
            self.drive_network = Network(model_name='Network', model_dir='/model_drive/')
        with g2.as_default():
            self.lanechange_network = Network(model_name='Lanechange', model_dir='/model_lanechange/')

        self._image_cut = image_cut
        self._image_size = (88, 200, 3)  # 아마 [세로, 가로, 차원(RGB)] 인듯?

        self.front_image = None
        self.object_detected_time = 0
        self.object_detected_duration = 0.7

        self._obstacle_far_ahead = False
        self._obstacle_far_detected_time = 0
        self._obstacle_far_detected_duration = 0.7

        # traffic light detection
        config_gpu = tf.ConfigProto()  # tf 설정 프로토콜인듯?
        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES
        config_gpu.gpu_options.visible_device_list = '0'  # GPU >= 2 일 때, 첫 번째 GPU만 사용

        self.return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                                "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        path = os.path.dirname(os.path.abspath(__file__))
        self.pb_file = os.path.join(path, "tensorflow_yolov3/yolov3_coco.pb")
        self.num_classes = 5
        self.traffic_image_input_size = 256
        self.tf_graph = tf.Graph()
        self.return_tensors = utils.read_pb_return_tensors(self.tf_graph, self.pb_file, self.return_elements)
        self.traffic_light_image = None
        self._is_traffic_light_in_distance = False
        self.bounding_boxes = None
        self.traffic_light_duration = 10.0
        self.traffic_light_detected_time = 0.0
        self.traffic_sess = tf.Session(graph=self.tf_graph, config=config_gpu)
        self.is_traffic_light_eu_style = True \
            if self._map.name.startswith("Town01") or self._map.name.startswith("Town02") else False

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

        # 벤치마크시 코스 이탈로 계속 정지해 있는 경우 강제로 움직이게 함
        c = self._vehicle.get_control()
        if c.steer == 0 and c.throttle == 0 and c.brake == 0:
            control.throttle = 1.0

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

        if direction == RoadOption.CHANGELANELEFT or direction == RoadOption.CHANGELANERIGHT:
            steer, acc, brake = self._compute_function(image_input, speed, direction, self.lanechange_network)
        else:
            steer, acc, brake = self._compute_function(image_input, speed, direction, self.drive_network)

        self.run_radar()

        self.traffic_light_detection()

        # a bit biased, but to reduce speed
        if self.speed >= 30:
            acc = acc * 0.5

        if self.bounding_boxes is not None and (len(self.bounding_boxes) > 0 and self.speed >= 5):
            acc = acc * 0.3

        if self._obstacle_far_ahead and self.speed >= 15:
            acc = acc * 0.3

        if self._obstacle_ahead or self.is_traffic_light_ahead():
            brake = 1.0
            acc = 0.0
        else:
            brake = 0.0

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
        speed = np.array(speed)

        speed = speed.reshape((1, 1))

        if control_input == RoadOption.LEFT:
            all_net = branches[0]
        elif control_input == RoadOption.RIGHT:
            all_net = branches[1]
        elif control_input == RoadOption.STRAIGHT:
            all_net = branches[2]
        elif control_input == RoadOption.LANEFOLLOW:
            all_net = branches[3]
        elif control_input == RoadOption.CHANGELANELEFT:
            all_net = branches[2]
        elif control_input == RoadOption.CHANGELANERIGHT:
            all_net = branches[3]
        else:
            all_net = branches[1]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(network.dropout_vec)}

        output_all = network.sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        return predicted_steers, predicted_acc, predicted_brake

    def set_radar_data(self, radar_data):
        self._radar_data = radar_data

    def set_stop_radar_range(self):
        hlc = self._local_planner.get_high_level_command()
        c = self._vehicle.get_control()
        sign = 1 if c.steer >= 0 else -1
        steer = abs(c.steer) * 100
        # 교차로 주행 시
        if hlc is RoadOption.RIGHT or hlc is RoadOption.LEFT:
            yaw_angle = 23
        elif hlc is RoadOption.STRAIGHT:
            yaw_angle = 20
        else:  # 교차로 아닌 경우
            yaw_angle = 12

        # right turn
        if sign > 0:
            left_offset = -steer * 0.7
            right_offset = steer
        else:
            left_offset = -steer
            right_offset = steer * 0.7

        return -(yaw_angle + left_offset), (yaw_angle + right_offset)

    def is_obstacle_ahead(self, _rotation, _detect, left_radar_range, right_radar_range):
        threshold = max(self.speed * 0.25, 2)
        if -7 <= _rotation.pitch <= 3 and left_radar_range <= _rotation.yaw <= right_radar_range and \
                0.5 * threshold < _detect.depth <= threshold:
            return True
        elif -7 <= _rotation.pitch <= 3 and left_radar_range - 5 <= _rotation.yaw <= right_radar_range + 5 and \
                0.3 * threshold < _detect.depth <= 0.5 * threshold:
            return True
        elif -7 <= _rotation.pitch <= 3 and left_radar_range - 15 <= _rotation.yaw <= right_radar_range + 15 and \
                _detect.depth <= 0.3 * threshold:
            return True
        return False

    def is_obstacle_far_ahead(self, _rotation, _detect):
        threshold = max(self.speed * 0.5, 10)
        if -3 <= _rotation.pitch <= 3 and -3 <= _rotation.yaw <= 3 \
                and _detect.depth <= threshold:
            return True
        return False

    def run_radar(self):
        if self._radar_data is None:
            return False

        if self.timestamp - self.traffic_light_detected_time > self.traffic_light_duration:
            self._is_traffic_light_in_distance = False

        if self.timestamp - self.object_detected_time > self.object_detected_duration:
            self._obstacle_ahead = False

        if self.timestamp - self._obstacle_far_detected_time > self._obstacle_far_detected_duration:
            self._obstacle_far_ahead = False

        current_rot = self._radar_data.transform.rotation

        left_radar_range, right_radar_range = self.set_stop_radar_range()
        for detect in self._radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)

            rotation = carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=azi,
                roll=current_rot.roll)

            if (not self.is_traffic_light_eu_style and (-10 <= azi <= 10 and 2 <= alt <= 15 and 20 <= detect.depth <= 40)) \
                    or (self.is_traffic_light_eu_style and 7 < azi < 12 and -4 < alt < 7 and 10 < detect.depth < 15):
                self._is_traffic_light_in_distance = True
                self.traffic_light_detected_time = self.timestamp
                break

            if self.is_obstacle_ahead(rotation, detect, left_radar_range, right_radar_range):
                self._obstacle_ahead = True
                self.object_detected_time = self.timestamp
                break

            if self.is_obstacle_far_ahead(rotation, detect):
                self._obstacle_far_ahead = True
                self._obstacle_far_detected_time = self.timestamp

    def traffic_light_detection(self):
        if self.is_traffic_light_eu_style:
            self.traffic_light_detection_eu()
        else:
            self.traffic_light_detection_us()

    def is_traffic_light_ahead(self):
        if self.is_traffic_light_eu_style:
            return self.is_traffic_light_ahead_eu()
        else:
            return self.is_traffic_light_ahead_us()

    def traffic_light_detection_us(self):
        if self.traffic_light_image is None:
            return

        image_cut = [0, 256, 272, 528]
        array = np.frombuffer(self.traffic_light_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.traffic_light_image.height, self.traffic_light_image.width, 4))

        frame_size = (self.traffic_image_input_size, self.traffic_image_input_size)
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]
        array = array[:, :, ::-1]

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        # 원하는 크기로 리사이즈
        image_pil = image_pil.resize((self.traffic_image_input_size, self.traffic_image_input_size))
        # image_pil.save('output/%06f.png' % time.time())
        np_image = np.array(image_pil, dtype=np.dtype("uint8"))

        # mask out side
        # left
        np_image[:, :int(self.traffic_image_input_size * 0.15)] = 0
        # right
        np_image[:, int(self.traffic_image_input_size * 0.8):self.traffic_image_input_size] = 0
        # top
        np_image[:int(self.traffic_image_input_size * 0.25)] = 0
        # bottom
        np_image[int(self.traffic_image_input_size * 0.67):] = 0

        image_raw = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preporcess(np.copy(image_raw),
                                            [self.traffic_image_input_size, self.traffic_image_input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.traffic_sess.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.traffic_image_input_size,
                                         score_threshold=0.6)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        self.bounding_boxes = bboxes

    def is_traffic_light_ahead_us(self):
        if self.bounding_boxes is None:
            return False

        if len(self.bounding_boxes) > 0 and self._is_traffic_light_in_distance and self._is_stop_line_ahead():
            return True
        else:
            return False

    def traffic_light_detection_eu(self):
        if self.traffic_light_image is None:
            return

        image_cut = [50, 306, 430, 716]

        array = np.frombuffer(self.traffic_light_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.traffic_light_image.height, self.traffic_light_image.width, 4))

        frame_size = (self.traffic_image_input_size, self.traffic_image_input_size)
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]
        array = array[:, :, ::-1]

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        # 원하는 크기로 리사이즈
        image_pil = image_pil.resize((self.traffic_image_input_size, self.traffic_image_input_size))
        # image_pil.save('output/%06f.png' % time.time())
        np_image = np.array(image_pil, dtype=np.dtype("uint8"))

        image_raw = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preporcess(np.copy(image_raw),
                                            [self.traffic_image_input_size, self.traffic_image_input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.traffic_sess.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.traffic_image_input_size,
                                         score_threshold=0.55)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        self.bounding_boxes = bboxes

    def is_traffic_light_ahead_eu(self):
        if self.bounding_boxes is None:
            return False

        if len(self.bounding_boxes) > 0 and \
                self._has_small_bounding_box() is False and self._is_traffic_light_in_distance:
            return True
        else:
            return False

    def _is_stop_line_ahead(self):
        if self.traffic_light_image is None:
            return
        threshold = 300 if self.speed < 10 else 0
        image_cut = [threshold, 600, 320, 480]

        array = np.frombuffer(self.traffic_light_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.traffic_light_image.height, self.traffic_light_image.width, 4))
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]  # 필요 없는 부분을 잘라내고
        array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        array = cv2.GaussianBlur(array, (3, 3), 0)
        array = cv2.Canny(array, 80, 120)

        return np.any(array > 0)

    def _has_small_bounding_box(self):
        for i, bbox in enumerate(self.bounding_boxes):
            coor = np.array(bbox[:4], dtype=np.int32)

            rect_size = (coor[2] - coor[0]) * (coor[3] - coor[1])

            if rect_size <= 3:
                return True
        return False
