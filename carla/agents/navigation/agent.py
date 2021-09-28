#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from enum import Enum

import carla
import math
from tqdm import tqdm
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._proximity_threshold = 10.0  # meters
        self._local_planner = None
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._last_traffic_light = None
        self._proximity_tlight_threshold = 10.0  # meters

        """
        for benchmark
        """
        self.collision_history = []

        self.red_light_violation_history = []
        self.red_light_history = []
        self.timestamp = 0

        self.is_traffic_light_eu_style = True \
            if self._map.name.startswith("Town01") or self._map.name.startswith("Town02") else False

        self.speed = 0
        self.bounding_boxes = None


    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()

        if debug:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

        return control

    def _is_light_red(self, lights_list, proximity_tlight_threshold):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        self._vehicle.get_transform(),
                                        proximity_tlight_threshold):
                if traffic_light.state == carla.TrafficLightState.Red or \
                        traffic_light.state == carla.TrafficLightState.Yellow:
                    return True, traffic_light

        return False, None

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    """
    for benchmark 
    """

    def add_red_light_history(self):
        log_stop = False
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        is_light_ahead, traffic_light = self._is_light_red(lights_list, self._proximity_tlight_threshold)

        if is_light_ahead:
            if len(self.red_light_history) < 1 or \
                    (len(self.red_light_history) > 0 and
                     traffic_light.id != self.red_light_history[-1].id):
                self.red_light_history.append(traffic_light)
                log_stop = True

        if log_stop:
            tqdm.write("red traffic light ahead")

    def is_violate_red_junction(self):
        if not self.is_traffic_light_eu_style:
            return self.is_violate_red_junction_us()
        else:
            return self.is_violate_red_junction_eu()

    def is_violate_red_junction_us(self):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if self._vehicle.is_at_traffic_light() and ego_vehicle_waypoint.is_junction:
            traffic_light = self._vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red and \
                    len(self.red_light_violation_history) < 1 or \
                    (len(self.red_light_violation_history) > 0 and
                     traffic_light.id != self.red_light_violation_history[-1].id):
                self.red_light_violation_history.append(traffic_light)
                return True

        return False

    def is_violate_red_junction_eu(self):
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        proximity_tlight_threshold = 3
        is_light_ahead, traffic_light = self._is_light_red(lights_list, proximity_tlight_threshold)

        if is_light_ahead:
            if len(self.red_light_violation_history) < 1 or \
                    (len(self.red_light_violation_history) > 0 and
                     traffic_light.id != self.red_light_violation_history[-1].id):
                self.red_light_violation_history.append(traffic_light)
                return True

        return False

    def is_off_course(self):
        next_waypoint = self._local_planner.get_next_waypoints()
        if next_waypoint is None:
            return False
        cur_loc = self._vehicle.get_location()
        dist = math.sqrt((next_waypoint.x - cur_loc.x) ** 2 + (next_waypoint.y - cur_loc.y) ** 2 +
                         (next_waypoint.z - cur_loc.z) ** 2)

        if dist > 10:
            return True
        else:
            return False


    def get_high_level_command(self, convert=True):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
            from agents.navigation.local_planner import RoadOption
            if _hcl == RoadOption.LEFT:
                return 1
            elif _hcl == RoadOption.RIGHT:
                return 2
            elif _hcl == RoadOption.STRAIGHT:
                return 3
            elif _hcl == RoadOption.LANEFOLLOW:
                return 4
            elif _hcl == RoadOption.CHANGELANELEFT:
                return 5
            elif _hcl == RoadOption.CHANGELANERIGHT:
                return 6

        # return self._local_planner.get_high_level_command()
        hcl = self._local_planner.get_high_level_command()
        if convert:
            return hcl_converter(hcl)
        else:
            return self._local_planner.get_high_level_command()

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()

