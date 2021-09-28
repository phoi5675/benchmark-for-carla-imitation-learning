"""
        Spawn NPCs into the simulation
        code from carla PythonAPI example/spawn_npc.py
"""
from carla import VehicleLightState as vls
import random
import logging
import time
import carla
from tqdm import tqdm


class SpawnNpc:
    def __init__(self):
        self.stop_requested = False
        self.vehicles_actor_list = []
        self.all_actors = None
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

    def spawn_npc(self, shared_stop_requested, number_of_vehicles=20, number_of_walkers=30):

        self.vehicles_actor_list.clear()
        self.vehicles_list.clear()
        self.walkers_list.clear()
        self.all_id.clear()

        synchronous_master = False
        random.seed(int(time.time()))

        try:
            client = carla.Client('127.0.0.1', 2000)
            client.set_timeout(10.0)

            world = client.get_world()

            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(1.0)

            blueprints = world.get_blueprint_library().filter('vehicle.*')
            blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

            blueprints = sorted(blueprints, key=lambda bp: bp.id)

            spawn_points = world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if number_of_vehicles < number_of_spawn_points:
                random.shuffle(spawn_points)
            elif number_of_vehicles > number_of_spawn_points:
                msg = 'requested %d vehicles, but could only find %d spawn points'
                logging.warning(msg, number_of_vehicles, number_of_spawn_points)
                number_of_vehicles = number_of_spawn_points

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            SetVehicleLightState = carla.command.SetVehicleLightState
            FutureActor = carla.command.FutureActor

            # --------------
            # Spawn vehicles
            # --------------
            batch = []
            for n, transform in enumerate(spawn_points):
                if n >= number_of_vehicles:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')

                # prepare the light state of the cars to spawn
                light_state = vls.NONE

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                             .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                             .then(SetVehicleLightState(FutureActor, light_state)))

            for response in client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                    logging.error(response.error)
                else:
                    self.vehicles_list.append(response.actor_id)

                '''
                actor = world.try_spawn_actor(blueprint, transform)
                if actor is None:
                    continue
                self.vehicles_actor_list.append(actor)
                self.vehicles_list.append(actor.id)
                '''
            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            percentagePedestriansRunning = 0.15  # how many pedestrians will run
            percentagePedestriansCrossing = 1.0  # how many pedestrians will walk through the road
            # 1. take all the random locations to spawn
            spawn_points = []
            for i in range(number_of_walkers):
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if loc != None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    self.walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(self.walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    self.walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(self.walkers_list)):
                self.all_id.append(self.walkers_list[i]["con"])
                self.all_id.append(self.walkers_list[i]["id"])
            self.all_actors = world.get_actors(self.all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            if not synchronous_master:
                world.wait_for_tick()
            else:
                world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ..])
            # set how many pedestrians can cross the road
            world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(self.all_id), 2):
                # start walker
                self.all_actors[i].start()
                # set walk to random point
                self.all_actors[i].go_to_location(world.get_random_location_from_navigation())
                # max speed
                self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

            tqdm.write('spawned %d vehicles and %d walkers.' % (len(self.vehicles_list), len(self.walkers_list)))

            # example of how to use parameters
            traffic_manager.global_percentage_speed_difference(30.0)

            while shared_stop_requested.value == 0:
                world.wait_for_tick()

        except Exception as e:
            logging.error(str(e))
        finally:
            shared_stop_requested.value = 0

            # print('\ndestroying %d vehicles' % len(self.vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].stop()

            # print('\ndestroying %d walkers' % len(self.walkers_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

            time.sleep(0.5)
