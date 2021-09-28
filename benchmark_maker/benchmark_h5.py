#!/usr/bin/env python

import h5py
import carla
import os
import numpy as np
from agents.navigation.local_planner import RoadOption


class BenchmarkH5:
    def __init__(self):
        self.start_transform = carla.Transform()
        self.end_transform = carla.Transform()
        self.route_length = 0.0
        self.has_turn = 0.0
        self.town_index = 0.0

        self.save_list = list()

    def save_h5_file(self, path='output/'):
        file_path = os.getcwd() + '/' + path

        # file name : data_#####.h5
        file_name = file_path + 'Town{0:02d}_all_routes.h5'.format(int(self.town_index))

        with h5py.File(file_name, 'w') as f:
            print(file_name)
            f.create_dataset('route', data=self.save_list, dtype=np.float32)
        print("saved %d files" % len(self.save_list))

    def route_to_h5_list(self, start_transform, end_transform, route_len):
        arr = (start_transform.location.x, start_transform.location.y, start_transform.location.z,
               start_transform.rotation.roll, start_transform.rotation.pitch, start_transform.rotation.yaw,
               end_transform.location.x, end_transform.location.y, end_transform.location.z,
               end_transform.rotation.roll, end_transform.rotation.pitch, end_transform.rotation.yaw,
               route_len, 1.0, self.town_index)
        self.save_list.append(arr)

    @staticmethod
    def has_turn(route):
        has_turn = False

        for hlc in route:
            if hlc[1] == RoadOption.RIGHT or hlc[1] == RoadOption.LEFT:
                has_turn = True
                break

        return has_turn
