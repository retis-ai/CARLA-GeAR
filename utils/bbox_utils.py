import carla
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Utils for debug drawing of bboxes
CONNECTIONS = [
	[1, 2], 
	[2, 4], 
	[4, 3], 
	[1, 3],
	[5, 6], 
	[6, 8], 
	[8, 7], 
	[7, 5], 
	[1, 5], 
	[2, 6], 
	[3, 7], 
	[4, 8]
]

COLOR_LABELS = {
    'car': [0, 0, 142/255],
    'motorcycle': [0, 0., 230/255],
    'truck': [0, 0., 70/255],
    'bicycle': [119/255, 11/255, 32/255],
    'pedestrian': [220/255, 20/255, 60/255],
    'traffic_light': [250/255, 170/255, 30/255],
    'stop_sign': [220/255, 220/255, 0]
}

# CAR_NAMES = ['vehicle']
BIKE_NAMES = ['Bike']
TRUCK_NAMES = ['Truck']
MOTORCYCLE_NAMES = ['Harley', 'Yamaha', 'Vespa', 'Kawasaki']
SIGN_NAMES = ['Stop']



    

class CameraParams():
    def __init__(self, image_size, fov):
        self.transform = carla.Transform()
        self.image_size = image_size
        self.fov = np.pi * fov /180
        self.camera_focal = 1/2 * self.image_size[1] / np.tan(self.fov/2)
        cpi = np.cos(np.pi/2)
        spi = np.sin(np.pi/2)
        self.T_cam_offset = np.transpose(np.array([[cpi, 0, spi, 0], [0, 1, 0, 0], [-spi, 0, cpi, 0], [0, 0, 0, 1]]) @ 
                            np.array([[cpi, spi, 0, 0], [-spi, cpi, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

        self.intrinsics = np.array([[self.camera_focal, 0, self.image_size[1]/2],
                    [0, self.camera_focal, self.image_size[0]/2], 
                    [0, 0, 1]])

        self.get_extrinsics(self.transform)

    def get_extrinsics(self, camera_transform):
        # print(camera_transform)
        self.transform = camera_transform
        self.camera_transform = camera_transform
        loc = np.array([[camera_transform.location.x], [-camera_transform.location.y], [camera_transform.location.z]])
        # ori = [np.pi / 180 * camera_transform.rotation.pitch, -np.pi / 180 * camera_transform.rotation.yaw, np.pi / 180 * camera_transform.rotation.roll]
        yaw = -np.pi / 180 * camera_transform.rotation.yaw

        # Compute camera transform
        camera_rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        T_WC_inv = np.block([[camera_rot_mat.T, -camera_rot_mat.T @ loc], [0, 0, 0, 1]])
        self.extrinsics = self.T_cam_offset @ T_WC_inv
        # print(self.extrinsics)
            


def get_camera_params(sensor_config):
    return CameraParams((sensor_config['height'], sensor_config['width']), sensor_config['fov'])
    

# Loop on actors and environment objects and collect all bounding boxes.
def get_world_bboxes(world):
    bboxes, transforms, labels, actors, orientations = [], [], [], [], []

    car_actors = {'actors': [actor for actor in world.get_actors() if 'vehicle' in actor.type_id], 'label': 'car'}
    walker_actors = {'actors': [actor for actor in world.get_actors() if 'walker' in actor.type_id], 'label': 'pedestrian'}
    actor_list = [car_actors, walker_actors]
    
    vehicle_objects = world.get_environment_objects(carla.CityObjectLabel.Vehicles)
    traffic_lights = {'objects': world.get_environment_objects(carla.CityObjectLabel.TrafficLight), 'label': 'traffic_light'}
    traffic_signs = {'objects': [obj for obj in world.get_environment_objects(carla.CityObjectLabel.TrafficSigns) if any([name in obj.name for name in SIGN_NAMES])], 'label': 'stop_sign'}

    # Filter vehicles (cars, trucks, bikes, motorbikes)
    bike_objects = {'objects': [obj for obj in vehicle_objects if any([name in obj.name for name in BIKE_NAMES])], 'label': 'bicycle'}
    moto_objects = {'objects': [obj for obj in vehicle_objects if any([name in obj.name for name in MOTORCYCLE_NAMES])], 'label': 'motorcycle'}
    truck_objects = {'objects': [obj for obj in vehicle_objects if any([name in obj.name for name in TRUCK_NAMES])], 'label': 'truck'}
    car_objects = {'objects': [obj for obj in vehicle_objects if obj not in bike_objects['objects'] + moto_objects['objects'] + truck_objects['objects']], 'label': 'car'}
    
    object_list = [bike_objects, moto_objects, truck_objects, car_objects, traffic_lights, traffic_signs]
    for obj_dict in object_list:
        for obj in obj_dict['objects']:
            # print(obj.name)
            bboxes.append(obj.bounding_box)
            orientations.append(obj.bounding_box.rotation.yaw)
            transforms.append(carla.Transform()) # if bbox is extracted from env obj, they are already in world coords
            labels.append(obj_dict['label'])
            actors.append(obj)
    
    for actor_dict in actor_list:
        for actor in actor_dict['actors']:
            # print(obj.name)
            bboxes.append(actor.bounding_box)
            orientations.append(actor.get_transform().rotation.yaw)
            transforms.append(actor.get_transform()) # if bbox is extracted from actor, they are in actor ref frame.
            labels.append(actor_dict['label'])
            actors.append(actor)
    
    return bboxes, transforms, labels, actors, orientations
        

# Complete bboxes computation pipeline.
def compute_bboxes(data, camera_params, world):
    if 'DepthCam' not in data.keys():
        return None, None, None, None, None, None
    h, w = camera_params.image_size

    # Get depth map and normalize
    depth_np = np.float32(np.frombuffer(data['DepthCam'][1].raw_data, dtype=np.uint8).reshape(h, w, 4)[:, :, :-1])
    depth_norm = 1000 * (depth_np[:, :, 2] + depth_np[:, :, 1] * 256 + depth_np[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    
    # Obtain 3D vertices in world
    bboxes, transforms, labels, actors, orientations = get_world_bboxes(world)
    vertices = [get_vertices_from_bbox(bbox, transform) for (bbox, transform) in zip(bboxes, transforms)]
    # orientations = [act.boundingtransform.rotation.yaw for act in actors]
    
    # Filter bboxes.
    idx_in_image = []
    vert_pix = []       # vertices on the image (in pixels)
    vert_cam = []       # vertices in the cam ref frame.
    bboxes_in_image = []
    orient = []
    for v_idx, (vert, label, actor, bbox, orientation) in enumerate(zip(vertices, labels, actors, bboxes, orientations)):
        vert_candidates, v_cam = project_vertices(vert, camera_params)
        if filter_bbox(vert_candidates, depth_norm):
            # print("OK")
            if 'stop' in label and not is_stop_visible(actor, camera_params):
                continue
            vert_pix.append(clip_vertices(vert_candidates, camera_params))
            idx_in_image.append(v_idx)
            vert_cam.append(v_cam)
            bboxes_in_image.append(bbox)
            orient.append(orientation)
    
    # Find 2d bounding boxes
    vert_2d = [convert_3d_vertices_to_2d(v) for v in vert_pix]
    labels = [labels[idx] for idx in idx_in_image]

    return bboxes_in_image, vert_pix, vert_2d, vert_cam, labels, orient



def get_vertices_from_bbox(bbox, transform):
    vert = bbox.get_world_vertices(transform) 
    vert = np.block([[v.x, -v.y, v.z, 1] for v in vert])
    return vert


# Return the positions of the 3D bbox in the image and distance from camera. 
def project_vertices(vert, camera_params):
    # Vertices in camera coords and projection on image.
    vert_cam = camera_params.extrinsics @ vert.T
    dist = vert_cam[2:3, :]
    vert_cam_ = vert_cam[:3] / dist
    vert_pix = camera_params.intrinsics @ vert_cam_
    vert_pix = vert_pix[:2, :].astype(np.int)
    return np.concatenate((vert_pix, dist), axis=0).T, vert_cam


def convert_3d_vertices_to_2d(vertices):
	# print(vertices)
	min_x, max_x, min_y, max_y = np.amin(vertices[:, 0]), np.amax(vertices[:, 0]), np.amin(vertices[:, 1]), np.amax(vertices[:, 1])
	# print(min_x, max_x, min_y, max_y)
	return np.array([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]])


# To clip bboxes that are outside the image.
def clip_vertices(vertices, camera_params):
    img_size = np.array([[camera_params.image_size[1], camera_params.image_size[0], 1000]])
    return np.clip(vertices, np.zeros((1, 3)), img_size)


# Consider only bboxes that are in the image, are not occluded, and are not too small.
def filter_bbox(vertices, depth_map):
    # Compute the midpoints of the "connections" between vertices. 
    midpoints = [(vertices[conn[0]-1, :] + vertices[conn[1]-1, :])/2 for conn in CONNECTIONS]
    # is_bbox_in_front = sum([is_point_in_front(mid) for mid in midpoints]) > 3
    is_bbox_in_image = sum([is_point_in_front(mid) and is_point_in_image(mid, depth_map.shape[:2]) and not is_point_occluded(mid, depth_map) for mid in midpoints]) > 3
    # is_bbox_occluded = sum([is_point_occluded(mid, depth_map) for mid in midpoints]) > 8

    # point_in_image = all(is_point_in_front(vertices[i, :]) and is_point_in_image(vertices[i, :], depth_map.shape[:2]) for i in range(vertices.shape[0])) and not is_bbox_small(vertices)
    # point_occluded = point_in_image and all(is_point_occluded(vertices[i, :], depth_map) for i in range(vertices.shape[0])) 
    return is_bbox_in_image and not is_bbox_small(vertices)


def is_stop_visible(actor, camera_params):
    stop_yaw = -actor.transform.rotation.yaw - 90
    camera_yaw = -camera_params.transform.rotation.yaw
    visual_angle = stop_yaw - camera_yaw
    visual_angle = (visual_angle + 360) % (360)
    # print("STOP:", stop_yaw, camera_yaw, visual_angle)
    return abs(visual_angle - 180) < 45


# False if vertex is behind camera
def is_point_in_front(point, max_dist=200):
    return point[2] > 0.5 and point[2] < max_dist


# False if vertex is outside camera sensor
def is_point_in_image(point, image_size):
    image_h, image_w = image_size
    return point[0] >= 0 and point[0] < image_w and point[1] >= 0 and point[1] < image_h


# Checks if point's distance is greater than depth map in that point (and its surroundings). 
# Bounding box is occluded if all vertices are occluded.
# TODO improve occlusion filtering: check if midpoints of connections are occluded, instead of vertices.
def is_point_occluded(point, depth_map):
    neighbors = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [0, 0]]
    occluded = []
    
    point_dist = point[-1]
    for n in neighbors:
        point_n = (point[:2] + np.array(n)).astype(np.int)
        occluded.append(False)
        if is_point_in_image(point_n, depth_map.shape[:2]) and point_dist > depth_map[point_n[1], point_n[0]]:
            # print(point_dist, depth_map[point_n[1], point_n[0]])
            occluded[-1] = True

    # print(occluded)
    return sum(occluded) > 4


# Checks if bbox has small area.
def is_bbox_small(vertices, min_area=900):
    w, h = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0]), np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
    return w * h < min_area

# Saves image with bounding boxes (debug)
def save_image_bboxes(filename, image, vertices, labels):
    plt.figure()
    plt.imshow(image)
    if len(vertices) > 0:
        is_2d = (vertices[0].shape[0] < 8)
        vert_connections = CONNECTIONS
        if is_2d:
            vert_connections = CONNECTIONS[:4]

        for v_idx, v in enumerate(vertices):
            for conn in vert_connections:
                plt.plot([v[conn[0]-1, 0], v[conn[1]-1, 0]], [v[conn[0]-1, 1], v[conn[1]-1, 1]], c=COLOR_LABELS[labels[v_idx]], linewidth=0.5) 
            plt.text(np.amin(v[:, 0]), np.amin(v[:, 1]), labels[v_idx], fontsize='xx-small', c=COLOR_LABELS[labels[v_idx]])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print("File %s saved." % filename)
    plt.close()