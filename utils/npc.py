import numpy as np
import random
import carla

class NPC:
    def __init__(self, world, verbose=False):
        self.world = world
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.vehicle_npcs = []
        self.pedestrian_npcs = []
        self.walker_controllers = []
        self.verbose = verbose

    def set_spawn_area(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def setup_vehicle_npcs(self, num_actors, vehicle_blueprints, client):

        waypoints = []
        for idx in range(num_actors):
            #if isinstance(self.min_x, tuple) and isinstance(self.max_x, tuple):
            #    x_range = random.choice([self.min_x, self.max_x])
            #    x = np.random.uniform(x_range[0], x_range[1], size=(1, ))[0]
            #else:
            x = np.random.uniform(self.min_x, self.max_x, size=(1,))[0]

            #if isinstance(self.min_y, tuple) and isinstance(self.max_y, tuple):
            #    y_range = random.choice([self.min_y, self.max_y])
            #    y = np.random.uniform(y_range[0], y_range[1], size=(1, ))[0]
            #else:
            y = np.random.uniform(self.min_y, self.max_y, size=(1,))[0]


            #location = random.choice(self.world.get_map().get_spawn_points())
            location = carla.Location(x=x, y=y, z=2.0)
            # print(location)
            wp = self.world.get_map().get_waypoint(location=location, project_to_road=True, lane_type=carla.LaneType.Driving) # | carla.LaneType.Shoulder)
            
            if wp is not None:
                # print("Changing z")
                # print(wp.transform.location.z)
                wp.transform.location.z = 2.0
                # print(wp.transform.location.z)
                waypoints.append(wp)
            
        num_actors = min(num_actors, len(waypoints))
        bps = np.random.choice(vehicle_blueprints, size=num_actors, replace=True)

        for bp, wp in zip(bps, waypoints):
            # wp.transform = 
            #npc = self.world.try_spawn_actor(bp, wp.transform)
            transf = carla.Transform(location=carla.Location(x=wp.transform.location.x, y=wp.transform.location.y, z=0.05), rotation=carla.Rotation(roll=wp.transform.rotation.roll, pitch=wp.transform.rotation.pitch, yaw=wp.transform.rotation.yaw))
            npc = self.world.try_spawn_actor(bp, transf)
            # print("Trying to spawn car")
            # print(wp.transform)
            if npc is not None:
                self.vehicle_npcs.append(npc)
                if self.verbose:
                    print("Spawned NPC car in (%.2f, %.2f, %.2f)" % (transf.location.x, transf.location.y, transf.location.z))

        # Moved in a different function
        # [vehicle.set_autopilot(enabled=True) for vehicle in self.vehicle_npcs]
        # self.world.tick()
        if self.verbose:
            print('created {} npc vehicles.'.format(len(self.vehicle_npcs)))


    def setup_pedestrian_npcs(self, num_actors):
        #walker_bps = np.random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'), size=num_actors, replace=True)
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        self.world.set_pedestrians_cross_factor(0.3)

        for idx in range(num_actors):
            bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            #location = carla.Location(x=x, y=y, z=2.0)
            # if isinstance(self.min_x, tuple) and isinstance(self.min_y, tuple):
            location = self.world.get_random_location_from_navigation() ## only to be used for finetuning_random!!
            # else:
            #     x = np.random.uniform(self.min_x, self.max_x, size=(1,))[0]
            #     y = np.random.uniform(self.min_y, self.max_y, size=(1,))[0]
            #     location = carla.Location(x=x, y=y, z=2.0)
            # wp = self.world.get_map().get_waypoint(location=location, project_to_road=True, lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder)    #carla.LaneType.Any)
            #transform = carla.Transform(location=wp.transform.location, rotation=carla.Rotation())
            #transform = carla.Transform(location=location, rotation=carla.Rotation())

            npc = None
            transf = carla.Transform(location, carla.Rotation(yaw=np.random.uniform(-90, 90)))
            # if wp is not None:
                # transf = wp.transform
            npc = self.world.try_spawn_actor(bp, transf) # waypoint.transform)

            #npc = self.world.try_spawn_actor(bp, transform) # waypoint.transform)

            if npc is not None:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=npc)
                self.pedestrian_npcs.append(npc)
                self.walker_controllers.append(controller)
                if self.verbose:
                    print("Spawned NPC walker in (%.2f, %.2f, %.2f)" % (transf.location.x, transf.location.y, transf.location.z))

        self.world.tick()
        
        # Moved in another function
        '''
        for controller in self.walker_controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())  # Between 1 and 2 m/s (default is 1.4 m/s).
        '''
        # self.world.tick()
        if self.verbose:
            print('created {} walkers.'.format(len(self.pedestrian_npcs)))

    def start_controller(self):
        for controller in self.walker_controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())  # Between 1 and 2 m/s (default is 1.4 m/s).

        [vehicle.set_autopilot(enabled=True) for vehicle in self.vehicle_npcs]
        for i in range(5):
            self.world.tick()

    def cleanup(self, client):
        client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in self.vehicle_npcs])
        if self.verbose:
            print('destroyed all vehicle npcs.')

        [controller.stop() for controller in self.walker_controllers]
        client.apply_batch([carla.command.DestroyActor(controller) for controller in self.walker_controllers])
        if self.verbose:
            print('destroyed all walker controllers.')

        client.apply_batch([carla.command.DestroyActor(pedestrian) for pedestrian in self.pedestrian_npcs])
        if self.verbose:
            print('destroyed all pedestrians.')

        self.vehicle_npcs = []
        self.pedestrian_npcs = []
        self.walker_controllers = []


