#!/usr/bin/env python


import h5py
import os
import random
import carla


class BenchmarH5Reader:
    def __init__(self):
        self.start_loc_x = "start_loc_x"
        self.start_loc_y = "start_loc_y"
        self.start_loc_z = "start_loc_z"
        self.start_rot_roll = "start_rot_roll"
        self.start_rot_pitch = "start_rot_pitch"
        self.start_rot_yaw = "start_rot_yaw"

        self.end_loc_x = "end_loc_x"
        self.end_loc_y = "end_loc_y"
        self.end_loc_z = "end_loc_z"
        self.end_rot_roll = "end_rot_roll"
        self.end_rot_pitch = "end_rot_pitch"
        self.end_rot_yaw = "end_rot_yaw"

        self.route_len = "route_len"
        self.has_turn = "has_turn"
        self.town_index = "town_index"
        self.h5_dict = [
            self.start_loc_x,
            self.start_loc_y,
            self.start_loc_z,
            self.start_rot_roll,
            self.start_rot_pitch,
            self.start_rot_yaw,
            self.end_loc_x,
            self.end_loc_y,
            self.end_loc_z,
            self.end_rot_roll,
            self.end_rot_pitch,
            self.end_rot_yaw,
            self.route_len,
            self.has_turn,
            self.town_index,
        ]

    @staticmethod
    def read_h5_file(town_index, path='route_data/'):
        file_path = os.getcwd() + '/' + path

        # file name : data_#####.h5
        file_name = file_path + 'Town{0:02d}_all_routes.h5'.format(int(town_index))

        all_routes = []
        with h5py.File(file_name, 'r') as f:
            route_dataset = f['route']

            for route in route_dataset:
                all_routes.append(route.astype(float))

        return all_routes

    @staticmethod
    def extract_test_list(all_route_list, num_of_test):
        route_list = []
        while len(route_list) < num_of_test:
            route_list.append(random.choice(all_route_list))

        return route_list

    def get_start_end_transform_from_h5_list(self, route):
        start_transform = carla.Transform(
            carla.Location(
                route[self.h5_dict.index(self.start_loc_x)],
                route[self.h5_dict.index(self.start_loc_y)],
                route[self.h5_dict.index(self.start_loc_z)]
            ),
            carla.Rotation(
                route[self.h5_dict.index(self.start_rot_roll)],
                route[self.h5_dict.index(self.start_rot_yaw)],
                route[self.h5_dict.index(self.start_rot_pitch)]
            )
        )
        end_transform = carla.Transform(
            carla.Location(
                route[self.h5_dict.index(self.end_loc_x)],
                route[self.h5_dict.index(self.end_loc_y)],
                route[self.h5_dict.index(self.end_loc_z)]
            ),
            carla.Rotation(
                route[self.h5_dict.index(self.end_rot_roll)],
                route[self.h5_dict.index(self.end_rot_yaw)],
                route[self.h5_dict.index(self.end_rot_pitch)]
            )
        )

        return start_transform, end_transform
