import carla
import time 
import queue

def carla_setup(town_name, SEED, debug=False):
# Connect to Carla Simulator
    count_errors = 0
    ok_connect = False
    while not ok_connect and count_errors < 3:
        try:
            client = carla.Client(host='localhost', port=2000)
            client.set_timeout(5.0)
            client.load_world(town_name)
        except:
            count_errors += 1
            time.sleep(1)
        finally:
            ok_connect = True
    
    synchronous_mode = True

    world = client.get_world()
    settings = world.get_settings()
    world.apply_settings(carla.WorldSettings(
        no_rendering_mode=not debug,
        synchronous_mode=synchronous_mode,
        fixed_delta_seconds=1/30,
    ))
    client.reload_world(False)

    weather_setting = carla.WeatherParameters.ClearNoon #   #CloudyNoon, ClearNoon, CloudySunset, ClearSunset, SoftRainNoon
    weather_setting.scattering_intensity = 0
    world.set_weather(weather_setting)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.set_synchronous_mode(synchronous_mode)
    traffic_manager.set_random_device_seed(SEED)
    return client, world, settings


def cleanup_env(ego, npc, client):
        time.sleep(1)
        ego.cleanup()
        # print('ego cleaned up.')
        npc.cleanup(client)
        # print('npc cleaned up.')


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


