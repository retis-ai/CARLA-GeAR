import yaml
import argparse
import os

import random
import numpy as np
import time

import carla
import json
import subprocess
import warnings

from utils.datasets_utils import set_dataset_paths, DataSaver, json_billboard_position
from utils.bbox_utils import get_camera_params, compute_bboxes
from utils.carla_utils import carla_setup, cleanup_env
from utils.ego_vehicle import EgoVehicle
from utils.npc import NPC


def collect_dataset(task, 
                    dataset_split, 
                    billboard_name, 
                    basepath, 
                    root_path, 
                    patch_type, 
                    net, 
                    patch_folder, 
                    town_name,
                    debug):
    from configs.simulation_configs import SEED_VALUES, ITERATION_VALUES, get_spawn_bp, BRANDS, MAX_VEHICLE_NPC, MAX_WALKER_NPC, patch_load_path
    # Set seeds and num_images.
    split_v = dataset_split.split('_')[0]
    SEED = SEED_VALUES[split_v]
    random.seed(SEED)
    np.random.seed(SEED)

    ITERATIONS = ITERATION_VALUES[split_v]
    STEP_ITERATIONS = 105 if split_v == 'video' else 1

    # TODO include additional info.
    debug_str = '=======================\nDATASET GENERATION INFO\nMain Folder: %s\nPatch Folder: %s\nTown Name: %s\n=========================='
    print(debug_str % (os.path.join(basepath, root_path), patch_folder, town_name))

    # Set image paths. dataset_paths is a dict with specific paths for each task.
    dataset_paths, sensors_config_file = set_dataset_paths(task, root_path, billboard_name, dataset_split)
    # leftImg8bit_path = os.path.join(root_path, billboard_name, 'leftImg8bit', dataset_split, dataset_split)
    # gtFine_path = os.path.join(root_path, billboard_name, 'gtFine', dataset_split, dataset_split)
    ds = DataSaver(task, basepath, dataset_paths)

    # The billboard config is dependent on town
    with open('configs/billboards_config_%s.yml' % town_name, 'r') as f:
        billboard_data = yaml.load(f)[billboard_name]

    with open(os.path.join('configs/sensors/', sensors_config_file), "r") as f:
        sensors_config = json.load(f)['sensors_config']
    
    # Carla setup.
    client, world, settings = carla_setup(town_name, SEED, debug=debug)
    
    # Check number of patches and number of billboards
    num_billboards = len(billboard_data)
    if 'no_billboard' in billboard_name:
        num_billboards = 0

    num_patches = 0
    
    if patch_folder is not None:
        patch_files = [os.path.join(patch_folder, p_file) for p_file in os.listdir(patch_folder) if 'png' in p_file]
        num_patches = len(patch_files)
        if num_patches > 2:
            raise Exception('A maximum of 2 patches are supported for multi-patch')
        elif num_patches > 1:
            if '_0.png' not in patch_files[0]:
                patch_files = patch_files[::-1]
                if '_0.png' not in patch_files[0]:
                    raise Exception('NO PATCH HAS 0 NUMBERING. Patches must be named ..._0.png and _1.png')
            if num_patches > num_billboards:
                warnings.warn('Number of billboards (%d) and number of patches (%d) does not match. Loading only allowed patches.' % (num_billboards, num_patches))
                num_patches = num_billboards


    # Find blueprint for patch spawning 
    billboard_bp = get_spawn_bp(billboard_name)
    if billboard_bp is not None:
        spawn_billboard_bp = world.get_blueprint_library().filter(billboard_bp)[0]
    # print(spawn_billboard_bp)

    billboards_transform = []

    # Spawn billboards and corresponding patches
    truck_actor = None
    if billboard_bp is not None:
        for i in range(num_billboards):
            os.system('rm %s' % patch_load_path)
            if i < num_patches:
                os.system('cp %s %s' % (patch_files[i], patch_load_path))
            if 'truck' not in billboard_name:
                
                bb_loc = billboard_data['b%d' % (i+1)]['location']
                bb_rot = billboard_data['b%d' % (i+1)]['rotation']
                
                loc = carla.Location(bb_loc['x'], bb_loc['y'], bb_loc['z'])
                rot = carla.Rotation(bb_rot['pitch'], bb_rot['yaw'], bb_rot['roll'])
                tr = carla.Transform(loc, rot)

                spawned_car = world.spawn_actor(spawn_billboard_bp, tr)

                world.tick()
                succ = spawned_car.destroy()

                billboards_transform.append(tr)

            else:
                truck_spawn_point = random.choice(world.get_map().get_spawn_points())
                truck_actor = world.spawn_actor(spawn_billboard_bp, truck_spawn_point)

    spawn_config = billboard_data[list(billboard_data.keys())[0]]['spawn_config']
    # Absolute positioning
    ego_min_x = spawn_config['ego']['min_x']
    ego_max_x = spawn_config['ego']['max_x']
    ego_min_y = spawn_config['ego']['min_y']
    ego_max_y = spawn_config['ego']['max_y']
    min_x = spawn_config['npc']['min_x']
    max_x = spawn_config['npc']['max_x']
    min_y = spawn_config['npc']['min_y']
    max_y = spawn_config['npc']['max_y']
    if billboard_data['b1']['location']['x'] is not None:
        x, y, z = billboards_transform[0].location.x, billboards_transform[0].location.y, billboards_transform[0].location.z 
        # Relative position wrt billboard01
        ego_min_x = x + spawn_config['ego']['min_x']
        ego_max_x = x + spawn_config['ego']['max_x']
        ego_min_y = y + spawn_config['ego']['min_y']
        ego_max_y = y + spawn_config['ego']['max_y']
        min_x = x + spawn_config['npc']['min_x']
        max_x = x + spawn_config['npc']['max_x']
        min_y = y + spawn_config['npc']['min_y']
        max_y = y + spawn_config['npc']['max_y']
    
    if 'no_billboard' in billboard_name or 'truck' in billboard_name:
        MAX_VEHICLE_NPC *= 100
        # MAX_WALKER_NPC *= 100

    vehicle_blueprints = []
    for brand in BRANDS:
        vehicle_blueprints.extend([bp for bp in world.get_blueprint_library().filter('vehicle.%s.*' % brand)])

    print('{} blueprints found for 4-wheeled vehicles.'.format(len(vehicle_blueprints))) #, vehicle_blueprints)

    camera_params = get_camera_params(sensors_config[0])
    camera_params_dx = get_camera_params(sensors_config[0])
    if sensors_config[1]['id'] == 'RGBCamDx':
        camera_params_dx = get_camera_params(sensors_config[1])

    # Meta information
    info = {
        'task': task,
        'num_billboards': num_billboards,
        'is_patched': patch_folder is not None, 
        'patch_folder': patch_folder,
        'patch_type': patch_type,
        'net': net,
        'creator': 'Federico Nesti',  # put your name here!
    }

    ego = EgoVehicle(world, sensors_config, verbose=debug)
    npc = NPC(world, verbose=debug)

    camera_billboard_positions = []
    if debug:
        spectator = world.get_spectator()
        times = []

    try:
        for cnt in range(ITERATIONS):
            print('Iteration no. %d / %d: ' % (cnt + 1, ITERATIONS))
            
            if debug:
                start_time = time.time()
            ego_spawn_point = random.choice(world.get_map().get_spawn_points())
            
            ego_random_point = carla.Location(x=np.random.uniform(ego_min_x, ego_max_x), y=np.random.uniform(ego_min_y, ego_max_y), z=1)
            if 'truck' in billboard_name:
                spawn_err_count = 0
                spawned = False
                # CHECK CAR SPAWN
                while not spawned and spawn_err_count < 5:
                    try:
                        ego_spawn_point = world.get_map().get_waypoint(location=ego_random_point, project_to_road=True, lane_type=carla.LaneType.Driving).transform
                        truck_distance = np.random.uniform(3, 30)
                        truck_transf = random.choice(world.get_map().get_waypoint(location=ego_spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving).next(truck_distance)).transform
                        truck_actor.set_transform(truck_transf)
                        billboards_transform = [truck_transf]
                        ego.setup_vehicle(vehicle_blueprints, ego_spawn_point)
                    except:
                        ego_random_point = carla.Location(x=np.random.uniform(ego_min_x, ego_max_x), y=np.random.uniform(ego_min_y, ego_max_y), z=1)
                        spawn_err_count += 1
                    else:
                        spawned = True

            elif 'no_billboard' not in billboard_name:
                
                ego_spawn_point = world.get_map().get_waypoint(location=ego_random_point, project_to_road=True, lane_type=carla.LaneType.Driving).transform

            # Spawn ego, then NPCs.
            if 'truck' not in billboard_name:
                ego.setup_vehicle(vehicle_blueprints, ego_spawn_point)

            if debug:
                spectator.set_transform(carla.Transform(carla.Location(z=2)+ego_spawn_point.location, ego_spawn_point.rotation))

            npc.set_spawn_area(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
            npc.setup_vehicle_npcs(num_actors=random.randint(0, MAX_VEHICLE_NPC), vehicle_blueprints=vehicle_blueprints, client=client) ## standard range (non-fintuning_random): 0 to 50
            npc.setup_pedestrian_npcs(num_actors=random.randint(0, MAX_WALKER_NPC)) ## standard range (non-fintuning_random): 0 to 20
            
            # Start autopilot now that everyone has been spawned.
            npc.start_controller()

            # Ego vehicle cameras setup
            ego.setup_sensors()
            ego.wait_for_sensors()

            
            for idx in range(STEP_ITERATIONS):   
                world.tick()        
                data = ego.get_sensor_data()
                
                # For stereo applications, sensors.json must have rgb at 1st place, rgb_dx at 2nd.
                camera_transform = ego._sensors_list[0].get_transform()
                camera_params.get_extrinsics(camera_transform)
                camera_transform_dx = ego._sensors_list[1].get_transform()
                camera_params_dx.get_extrinsics(camera_transform_dx)
                camera_params_list = [camera_params, camera_params_dx]

                bboxes_in_image, vert_pix, vert_2d, vert_cam, labels, orientations = compute_bboxes(data, camera_params, world)

                # Save Images
                ds.save_image(data)

                # Accumulate billboard positions.
                camera_billboard_positions.append(json_billboard_position(billboards_transform, [camera_transform, camera_transform_dx], ds.img_count))

                ds.save_annotations(vert_2d, vert_cam, bboxes_in_image, labels, camera_params_list, split, orientations)

                

            cleanup_env(ego=ego, npc=npc, client=client)
        
            if debug:
                times.append(time.time() - start_time)

    finally:
        # Save camera and billboard positions.
        ds.save_camera_positions(info, camera_billboard_positions, camera_params_list)
        # Clean up Carla world!
        world.apply_settings(settings)
        cleanup_env(ego=ego, npc=npc, client=client)
        if truck_actor is not None:
            truck_actor.destroy()
        
        if debug:
            print("Timing analysis: mean %.2f sec/img, std %.2f" % (np.array(times).mean(), np.array(times).std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/collection_config.yml',
        help="config file (yml format)",
    )


    args = parser.parse_args()
    cfg_file = args.config

    with open(cfg_file, "r") as f:
        cfg = yaml.load(f)

    print("CFG", cfg)
    root = cfg['root']
    basepath = cfg['basepath']

    town_name = cfg['town_name']
    # weather: # TODO

    task = cfg['task']
    split = cfg['split']
    billboard = cfg['billboard'] 
    patch_folder = cfg['patch_folder']

    patch_type = cfg['patch_type']
    net = cfg['net']

    debug = cfg['debug']

    if task not in ('ss', '2dod', '3dod', 'depth'):
        raise Exception('Task specified is not supported. Choose one between ss, 2dod, 3dod, depth')


    collect_dataset(task=task, dataset_split=split, 
                        billboard_name=billboard, 
                        basepath=basepath, 
                        root_path=root, 
                        patch_type=patch_type, 
                        net=net, 
                        patch_folder=patch_folder, 
                        town_name=town_name,
                        debug=debug)
    # To avoid early resets in automatic data collection.
    time.sleep(10)
