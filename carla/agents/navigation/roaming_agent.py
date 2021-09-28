#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner


class RoamingAgent(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlanner(self._vehicle)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        self._state = AgentState.NAVIGATING
        # standard local planner behavior
        control = self._local_planner.run_step(debug=debug)

        if hazard_detected:
            control = self.emergency_stop()

        return control

    # ====================================================================
    # ----- appended from original code ----------------------------------
    # ====================================================================
    def get_high_level_command(self):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
            from agents.navigation.local_planner import RoadOption
            REACH_GOAL = 0.0
            GO_STRAIGHT = 5.0
            TURN_RIGHT = 4.0
            TURN_LEFT = 3.0
            LANE_FOLLOW = 2.0

            if _hcl == RoadOption.STRAIGHT:
                return GO_STRAIGHT
            elif _hcl == RoadOption.LEFT:
                return TURN_LEFT
            elif _hcl == RoadOption.RIGHT:
                return TURN_RIGHT
            elif _hcl == RoadOption.LANEFOLLOW or _hcl == RoadOption.VOID:
                return LANE_FOLLOW
            else:
                return REACH_GOAL

        hcl = self._local_planner.get_high_level_command()
        return hcl_converter(hcl)

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()
