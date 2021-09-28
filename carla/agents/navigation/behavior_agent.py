# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
import time
import math
from agents.navigation.agent import Agent
from agents.navigation.local_planner_behavior import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.types_behavior import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive


class BehaviorAgent(Agent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment,
    such as overtaking or tailgating avoidance. Adding to these are possible
    behaviors, the agent can also keep safety distance from a car in front of it
    by tracking the instantaneous time to collision and keeping it in a certain range.
    Finally, different sets of behaviors are encoded in the agent, from cautious
    to a more aggressive ones.
    """

    def __init__(self, vehicle, ignore_traffic_light=False, behavior='normal'):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param ignore_traffic_light: boolean to ignore any traffic light
            :param behavior: type of agent to apply
        """

        super(BehaviorAgent, self).__init__(vehicle)
        self.vehicle = vehicle
        self.ignore_traffic_light = ignore_traffic_light
        self._local_planner = LocalPlanner(self)
        self._grp = None
        self.look_ahead_steps = 0

        # Vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.is_at_traffic_light = 0
        self.light_state = "Green"
        self.light_id_to_ignore = -1
        self.min_speed = 5
        self.behavior = None
        self._sampling_resolution = 4.5

        # Parameters for agent behavior
        if behavior == 'cautious':
            self.behavior = Cautious()

        elif behavior == 'normal':
            self.behavior = Normal()

        elif behavior == 'aggressive':
            self.behavior = Aggressive()

        self._radar_data = None
        self._obstacle_ahead = False

        self.noise_steer = 0
        self.noise_steer_max = 0
        self.noise_start_time = 0
        self.noise_active_duration = 0
        self.is_noise_increase = True
        self.noise_bias = 0
        self.noise_duration = 7
        self.steer = 0

        self.weird_steer_count = 0
        self.weird_reset_count = 0

    def update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self.speed = get_speed(self.vehicle)
        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW

        self.look_ahead_steps = int(self.speed_limit / 10)

        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

        self.is_at_traffic_light = self.vehicle.is_at_traffic_light()
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            # This method also includes stop signs and intersections.
            self.light_state = str(self.vehicle.get_traffic_light_state())

    def set_destination(self, start_location, end_location, clean=False):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router.

            :param start_location: initial position
            :param end_location: final position
            :param clean: boolean to clean the waypoint queue
        """
        if clean:
            self._local_planner.waypoints_queue.clear()
        self.start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)

        self._local_planner.set_global_plan(route_trace, clean)

        self._local_planner.change_intersection_hcl()

        self.weird_steer_count = 0
        self.weird_reset_count = 0

        print("set new waypoint")

    def reroute(self, spawn_points):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.

            :param spawn_points: list of possible destinations for the agent
        """

        print("Target almost reached, setting new destination...")
        random.shuffle(spawn_points)
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        destination = spawn_points[0].location if spawn_points[0].location != new_start else spawn_points[1].location
        print("New destination: " + str(destination))

        self.set_destination(new_start, destination)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        # Setting up global router
        if self._grp is None:
            wld = self.vehicle.get_world()
            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def traffic_light_manager(self, waypoint):
        """
        This method is in charge of behaviors for red lights and stops.

        WARNING: What follows is a proxy to avoid having a car brake after running a yellow light.
        This happens because the car is still under the influence of the semaphore,
        even after passing it. So, the semaphore id is temporarely saved to
        ignore it and go around this issue, until the car is near a new one.

            :param waypoint: current waypoint of the agent
        """

        light_id = self.vehicle.get_traffic_light().id if self.vehicle.get_traffic_light() is not None else -1

        if self.light_state == "Red":
            if not waypoint.is_junction and (self.light_id_to_ignore != light_id or light_id == -1):
                return 1
            elif waypoint.is_junction and light_id != -1:
                self.light_id_to_ignore = light_id
        if self.light_id_to_ignore != light_id:
            self.light_id_to_ignore = -1
        return 0

    def _overtake(self, location, waypoint, vehicle_list):
        """
        This method is in charge of overtaking behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        if (left_turn == carla.LaneChange.Left or left_turn ==
            carla.LaneChange.Both) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1)
            if not new_vehicle_state:
                print("Overtaking to the left!")
                self.behavior.overtake_counter = 200
                self.set_destination(left_wpt.transform.location,
                                     self.end_waypoint.transform.location, clean=True)
        elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1)
            if not new_vehicle_state:
                print("Overtaking to the right!")
                self.behavior.overtake_counter = 200
                self.set_destination(right_wpt.transform.location,
                                     self.end_waypoint.transform.location, clean=True)

    def _tailgating(self, location, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
            self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self.speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    self.behavior.tailgate_counter = 200
                    self.set_destination(right_wpt.transform.location,
                                         self.end_waypoint.transform.location, clean=True)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    self.behavior.tailgate_counter = 200
                    self.set_destination(left_wpt.transform.location,
                                         self.end_waypoint.transform.location, clean=True)

    def collision_and_car_avoid_manager(self, location, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible overtaking or tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self.vehicle.id]

        if self.direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=30)

            # Check for overtaking

            if vehicle_state and self.direction == RoadOption.LANEFOLLOW and \
                    not waypoint.is_junction and self.speed > 10 \
                    and self.behavior.overtake_counter == 0 and self.speed > get_speed(vehicle):
                self._overtake(location, waypoint, vehicle_list)

            # Check for tailgating

            elif not vehicle_state and self.direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self.speed > 10 \
                    and self.behavior.tailgate_counter == 0:
                self._tailgating(location, waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, location, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        def dist(w):
            return w.get_location().distance(waypoint.transform.location)

        walker_list = [w for w in walker_list if dist(w) < 10]

        if self.direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self.speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self.behavior.safety_time > ttc > 0.0:
            control = self._local_planner.run_step(
                target_speed=min(positive(vehicle_speed - self.behavior.speed_decrease),
                                 min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)),
                debug=debug)
        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self.behavior.safety_time > ttc >= self.behavior.safety_time:
            control = self._local_planner.run_step(
                target_speed=min(max(self.min_speed, vehicle_speed),
                                 min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)),
                debug=debug)
        # Normal behavior.
        else:
            control = self._local_planner.run_step(
                target_speed=min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

        return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        control = None
        if self.behavior.tailgate_counter > 0:
            self.behavior.tailgate_counter -= 1
        if self.behavior.overtake_counter > 0:
            self.behavior.overtake_counter -= 1

        ego_vehicle_loc = self.vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # check maneuvering
        c = self._vehicle.get_control()

        if self.is_maneuvering_weird(c):
            self.weird_steer_count += 1

        if self.weird_steer_count >= 10:
            print("vehicle is steering in wrong way")
            self.weird_steer_count = 0
            self.weird_reset_count += 1

        if self.weird_reset_count > 2:
            self.reset_destination()

        control = self._local_planner.run_step(target_speed=self.speed_limit, debug=debug)

        if self.noise_steer != 0:
            signed = -1 if self.noise_steer > 0 else 1
            control.steer = abs(control.steer) * (random.uniform(1.1, 1.3) * signed)

        self.steer = control.steer
        return control

    # ====================================================================
    # ----- appended from original code ----------------------------------
    # ====================================================================
    def get_high_level_command(self):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
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

        return self._local_planner.get_high_level_command().value
        # hcl = self._local_planner.get_high_level_command()
        # return hcl_converter(hcl)

    def is_maneuvering_weird(self, control):
        hcl = self._local_planner.get_high_level_command()
        turn_threshold = 0.7
        if hcl == RoadOption.STRAIGHT or hcl == RoadOption.LANEFOLLOW:
            turn_threshold = 0.5
            if abs(control.steer) >= turn_threshold:
                return True
        elif (hcl == RoadOption.LEFT or hcl == RoadOption.CHANGELANELEFT) and control.steer >= turn_threshold:
            return True
        elif (hcl == RoadOption.RIGHT or hcl == RoadOption.CHANGELANERIGHT) and control.steer <= -turn_threshold:
            return True
        elif (hcl == RoadOption.CHANGELANERIGHT or hcl == RoadOption.CHANGELANELEFT) \
                and abs(control.steer) >= turn_threshold:
            return True
        else:
            return False

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()

    def is_dest_far_enough(self):
        return self._local_planner.is_dest_far_enough()

    def set_radar_data(self, radar_data):
        self._radar_data = radar_data

    def set_stop_radar_range(self):
        hcl = self._local_planner.get_high_level_command()
        c = self._vehicle.get_control()
        steer = abs(c.steer) * 20
        # 교차로 주행 시
        if hcl is RoadOption.RIGHT or hcl is RoadOption.LEFT or hcl is RoadOption.STRAIGHT:
            yaw_angle = 40
        else:  # 교차로 아닌 경우
            yaw_angle = 15
        return yaw_angle + steer

    def set_target_speed(self, speed):
        self._local_planner.set_target_speed(speed)

    def is_obstacle_ahead(self, _rotation, _detect):
        radar_range = self.set_stop_radar_range()

        threshold = max(self._speed * 0.2, 3)
        if -5.0 <= _rotation.pitch <= 5.0 and -radar_range <= _rotation.yaw <= radar_range and \
                _detect.depth <= threshold:
            return True
        return False

    def is_obstacle_far_ahead(self, _rotation, _detect):
        radar_range = self.set_stop_radar_range()
        c = self._vehicle.get_control()
        steer = abs(c.steer) * 4
        left_margin = steer if c.steer < 0 else 0
        right_margin = steer if c.steer > 0 else 0
        if 1.0 <= _rotation.pitch <= 5.0 and -(5 + left_margin) <= _rotation.yaw <= (5 + right_margin) \
                and _detect.depth <= 15:
            return True
        return False

    def run_radar(self):
        if self._radar_data is None:
            return False
        current_rot = self._radar_data.transform.rotation
        self._obstacle_ahead = False

        for detect in self._radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)

            rotation = carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=azi,
                roll=current_rot.roll)

            if self.is_obstacle_ahead(rotation, detect):
                self._obstacle_ahead = True

    def reset_destination(self):
        sp = self._map.get_spawn_points()
        rand_sp = random.choice(sp)
        self._vehicle.set_transform(rand_sp)

        control_reset = carla.VehicleControl()
        control_reset.steer, control_reset.throttle, control_reset.brake = 0.0, 0.0, 0.0
        self._vehicle.apply_control(control_reset)

        spawn_point = random.choice(self._map.get_spawn_points())
        self.set_destination(start_location=carla.Location(rand_sp.location.x, rand_sp.location.y, rand_sp.location.z),
                             end_location=carla.Location(spawn_point.location.x, spawn_point.location.y, spawn_point.location.z),
                             clean=True)

        self.weird_reset_count = 0
        self.weird_steer_count = 0

    def noisy_agent(self):
        cur_time = time.time()
        self.noise_active_duration = self.noise_duration * random.uniform(0.2, 0.3)
        if self.noise_start_time == 0:
            signed = random.choice([-1, 1])
            self.noise_start_time = cur_time
            self.noise_steer_max = random.uniform(0.2, 0.5) * signed
            self.noise_steer = 0.1 * signed
            self.noise_bias = random.uniform(0.03, 0.05) * signed
        elif cur_time - self.noise_start_time > self.noise_duration:
            self.noise_start_time = 0
            self.noise_steer_max = 0
            self.noise_steer = 0
            self.noise_bias = 0
            self.is_noise_increase = True
        elif cur_time - self.noise_start_time < self.noise_active_duration:
            if abs(self.noise_steer) < abs(self.noise_steer_max) and self.is_noise_increase:
                self.noise_steer += self.noise_bias
            elif abs(self.noise_steer) > abs(self.noise_steer_max) * 0.9 or self.is_noise_increase is False:
                self.is_noise_increase = False
                self.noise_steer -= self.noise_bias
            if abs(self.noise_steer) < abs(self.noise_bias) * 1.1:
                self.noise_steer = 0
            return self.noise_steer
        elif self.noise_active_duration < cur_time - self.noise_start_time < self.noise_duration:
            return 0.0
        return 0.0
