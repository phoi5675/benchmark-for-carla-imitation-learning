import datetime


class ResultSaver:
    def __init__(self):
        self.weather = ''
        self.town = ''
        self.agent_name = ''
        self.is_dynamic = False
        self.total_run = 0
        self.success = 0
        self.timeout = 0
        self.stuck = 0
        self.collision_car = 0
        self.collision_ped = 0
        self.collision_other = 0
        self.red_light_count = 0
        self.traffic_light_violation = 0
        self.traffic_light_no_bboxes = 0
        self.distance_traveled = 0

        self.res_str = ''

        self.res_str_tlight_vio = ''
        self.weather_tlight_vio = {}

    def make_res(self):
        self.res_str += "result for agent : %s, map : %s, weather : %s, dynamic : %s\n" \
                        % (self.agent_name, self.town, self.weather, self.is_dynamic)
        self.res_str += "total\tsuccess\ttimeout\tstuck\tcol_car\tcol_ped\tcol_oth\tlht_vio\tno_bbx\ttotrdlt\ttraveled(m)\n"
        self.res_str += "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\n" \
                        % (self.total_run, self.success, self.timeout, self.stuck, self.collision_car,
                           self.collision_ped, self.collision_other, self.traffic_light_violation,
                           self.traffic_light_no_bboxes, self.red_light_count, self.distance_traveled)
        self.res_str_tlight_vio += "traffic light violation details\n"
        self.res_str_tlight_vio += "weather\t\ttlht_vio\tno_bbx\ttotrdlt\n"
        for k in self.weather_tlight_vio.keys():
            self.res_str_tlight_vio += "%s\t" % k
            for n in self.weather_tlight_vio[k]:
                self.res_str_tlight_vio += "%d\t" % n
            self.res_str_tlight_vio += "\n"

    def show_res(self):
        print(self.res_str)
        print(self.res_str_tlight_vio)

    def save_res(self):
        date = datetime.datetime.now()
        now = date.strftime('%Y-%m-%d_%H%M')
        dynamic = 'dynamic' if self.is_dynamic else 'no_dynamic'

        file = open("result/" + self.agent_name + "_" + self.weather + "_" + dynamic + "_" +
                    self.town + "_" + now + " benchmark.txt", "w")
        file.write(self.res_str)
        file.write(self.res_str_tlight_vio)

        file.close()

    def calc_collision(self, agent):
        for event in agent.collision_history:
            type_id = event.other_actor.type_id
            if type_id.startswith("vehicle"):
                self.collision_car += 1
            elif type_id.startswith("walker"):
                self.collision_ped += 1
            else:
                self.collision_other += 1

    def is_dynamic_env(self, vehicles, walkers):
        self.is_dynamic = True if vehicles > 1 and walkers > 1 else False
