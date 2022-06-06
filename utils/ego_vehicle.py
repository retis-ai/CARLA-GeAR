import random
import time

import carla
from .sensor_interface import CallBack, SensorInterface

class EgoVehicle:
    def __init__(self, world, sensors_config, verbose=False):
        self.world = world
        self.start_transform = None

        self.sensor_interface = SensorInterface()
        self.sensors = sensors_config
        self._sensors_list = []
        self.vehicle = None
        self.verbose = verbose

    def setup_sensors(self):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in self.sensors:
            # These are the sensors spawned on the carla world
            bp = bp_library.find(str(sensor_spec['type']))
            if sensor_spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'], roll=sensor_spec['roll'], yaw=sensor_spec['yaw'])
            
            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = self.world.spawn_actor(bp, sensor_transform, attach_to=self.vehicle)
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor, self.sensor_interface))
            self._sensors_list.append(sensor)

        # Tick once to spawn the sensors
        self.world.tick()

    def setup_vehicle(self, vehicle_blueprints, start_transform=None):
        self.start_transform = start_transform
        if start_transform is None:
            self.start_transform = random.choice(self.world.get_map().get_spawn_points())

        self.start_transform.location.z = 0.1 # need to raise vehicle to avoid collision at spawn time
        
        ## needed for billboard 02
        #self.start_transform.rotation.pitch=0.0
        #self.start_transform.rotation.yaw=0.0
        

        if self.vehicle is None:
            
            bp = random.choice(vehicle_blueprints)
            # sprinter dimenstions raise problems with camera config.
            while 'sprinter' in bp.id:
                bp = random.choice(vehicle_blueprints)

            self.vehicle = None
            count_try = 0
            while self.vehicle is None and count_try < 10:
                self.vehicle = self.world.try_spawn_actor(bp, self.start_transform)
                self.start_transform.location.z += 0.1
                count_try += 1

            if count_try >= 10:
                raise Exception("Impossible to spawn ego vehicle. It's possible that the spawn config is wrong?")

        if self.vehicle is not None:
            if self.verbose:
                print('Spawned Ego Vehicle at (%.2f, %.2f, %.2f): ' % (start_transform.location.x, start_transform.location.y, start_transform.location.z))
        
        self.vehicle.set_autopilot(enabled=True)
        self.world.tick()


    def get_sensor_data(self):
        return self.sensor_interface.get_data()

    def wait_for_sensors(self, num_ticks=25):
        for _ in range(num_ticks):
            self.sensor_interface.get_data()
            self.world.tick()

        if self.verbose:
            print('all sensors ready to go...')

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                if self.verbose:
                    print("Destroying: ", self._sensors_list[i])
                self._sensors_list[i].stop()
                time.sleep(1)
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []

        self.sensor_interface.reset()
        if self.verbose:
            print('destroyed sensors.')

        if self.vehicle is not None:
            self.vehicle.set_autopilot(False)
            self.vehicle.destroy()
        self.vehicle = None

        if self.verbose:
            print('destroyed ego.')



