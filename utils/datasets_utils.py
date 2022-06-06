import numpy as np
import functools
import os
import imageio
from .sstags_carla_to_cityscapes import convert_single_image_to_cityscapes
# from .bbox_utils import get_world_bboxes, get_vertices_from_bbox, project_vertices, filter_bbox, is_stop_visible, clip_vertices
import carla
import datetime
import json



class DataSaver():
    '''
    This class takes the data and saves it according to the data structure corresponding to the task.
    '''

    def __init__(self, task, basepath, dataset_paths):
        self.task = task
        self.dataset_paths = dataset_paths
        key2savedata = {
            'ss': self._save_ss_data,
            '3dod': self._save_3dod_data,
            '2dod': self._save_2dod_data,
            'depth': self._save_depth_data
        }

        key2saveann = {
            'ss': self._save_ss_ann,
            '3dod': self._save_3dod_ann,
            '2dod': self._save_2dod_ann,
            'depth': self._save_depth_ann
        }

       
        self.save_image = key2savedata[self.task]
        self.save_annotations = key2saveann[self.task]
        self.basepath = basepath

        self.img_count = 0
        self.obj_count = 0

        # COCO stuff
        self.annotations = []
        self.images = []
        self.info = {
            "year": 2022,
            "version": '0.0',
            "description": 'COCO-like dataset for CARLA images',
            "contributor": "Federico Nesti",
            "url": '',
            "date_created": datetime.datetime.now()
        }
        self.licenses = []
        self.filelist = []
        
        # Create dirs for image and annotations.
        for path in dataset_paths.values():
            if not os.path.isdir(os.path.join(basepath, path)):
                os.makedirs(os.path.join(basepath, path))

    # Task-specific images save functions
    def _save_ss_data(self, ss_data):
        rgb_filename = os.path.join(self.basepath, self.dataset_paths['images'], 'rgb_%04d.png' % self.img_count)
        ss_data['RGBCam'][1].save_to_disk(rgb_filename, carla.ColorConverter.Raw)
        ss_filename = os.path.join(self.basepath, self.dataset_paths['labels'], 'rgb_%04d.png' % self.img_count)
        ss_data['SSCam'][1].save_to_disk(ss_filename, carla.ColorConverter.CityScapesPalette)
        
        # convert to cityscapes Tags.
        image = imageio.imread(ss_filename)
        image = image[:, :, :3] # convert from Carla's RGBA format to RGB format

        cityscapes_color_image, cityscapes_label_image = convert_single_image_to_cityscapes(image)

        ss_color_filename = ss_filename[:-4] + '_gtFine_color.png'
        ss_id_filename = ss_filename[:-4] + '_gtFine_labelIds.png'
        imageio.imwrite(ss_color_filename, cityscapes_color_image) # output filename format for RGB = aachen_000053_000019_gtFine_color.png
        imageio.imwrite(ss_id_filename, cityscapes_label_image) # output filename format for LabelID = aachen_000053_000019_gtFine_labelIds.png 

    def _save_3dod_data(self, data_3dod):
        rgb_filename = os.path.join(self.basepath, self.dataset_paths['images'], '%06d.png' % self.img_count)
        data_3dod['RGBCam'][1].save_to_disk(rgb_filename, carla.ColorConverter.Raw)
        rgb_dx_filename = os.path.join(self.basepath, self.dataset_paths['images_dx'], '%06d.png' % self.img_count)
        data_3dod['RGBCamDx'][1].save_to_disk(rgb_dx_filename, carla.ColorConverter.Raw)
        self.filelist.append('%06d\n' % self.img_count)

    def _save_2dod_data(self, data_2dod):
        rgb_filename = os.path.join(self.basepath, self.dataset_paths['images'], 'rgb_%04d.png' % self.img_count)
        data_2dod['RGBCam'][1].save_to_disk(rgb_filename, carla.ColorConverter.Raw)
        image = imageio.imread(rgb_filename)
        self.images.append(coco_format_image(self.img_count, image.shape[:2], os.path.basename(rgb_filename)))

    def _save_depth_data(self, depth_data):
        rgb_filename = os.path.join(self.basepath, self.dataset_paths['images'], 'rgb_%04d.png' % self.img_count)
        depth_data['RGBCam'][1].save_to_disk(rgb_filename, carla.ColorConverter.Raw)
        depth_filename = os.path.join(self.basepath, self.dataset_paths['labels'], 'rgb_%04d.png' % self.img_count)
        depth_data['DepthCam'][1].save_to_disk(depth_filename, carla.ColorConverter.Depth)
        filename_path = os.path.join(*self.dataset_paths['images'].split('/')[-2:])
        labels_path = os.path.join(*self.dataset_paths['labels'].split('/')[-2:])
        self.filelist.append(os.path.join('/', filename_path, 'rgb_%04d.png' % self.img_count) + \
            ' /' + os.path.join(labels_path, 'rgb_%04d.png' % self.img_count) + '\n')
    

    # Task-specific annotation save functions
    def _save_ss_ann(self, vert_2d, vert_cam, bboxes_in_image, labels, camera_params_list, dataset_split, orientations):
        # Data already saved as image. Increment counter.
        self.img_count += 1

    def _save_3dod_ann(self, vert_2d, vert_cam, bboxes_in_image, labels, camera_params_list, dataset_split, orientations):
        camera_params, camera_params_dx = camera_params_list
        for (vert, v_cam, bbox, label, orient) in zip(vert_2d, vert_cam, bboxes_in_image, labels, orientations):
            self.annotations.append(kitti_format_annotation(bbox, vert, v_cam, label, camera_params.transform, orient))
        # Add dummy label (to write a file)
        if not self.annotations:
            self.annotations.append('%s %.2f %d %.2f %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % ('DontCare', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        annotation_filename = '%06d.txt' % self.img_count
        with open(os.path.join(self.basepath, self.dataset_paths['labels'], annotation_filename), "w") as f:
            for ann in self.annotations:
                f.write(ann)
        
        with open(os.path.join(self.basepath, self.dataset_paths['calib'], annotation_filename), "w") as f:
            f.write(kitti_format_calib(camera_params, camera_params_dx))
        
        self.obj_count += 1
        self.img_count += 1
        self.annotations = []
        filelist = ''.join(self.filelist)[:-1]
        with open(os.path.join(self.basepath, self.dataset_paths['splits'], '%s.txt' % dataset_split), "w") as f:
            f.write(filelist)

    def _save_2dod_ann(self, vert_2d, vert_cam, bboxes_in_image, labels, camera_params_list, dataset_split, orientations):
        for v_idx, vert in enumerate(vert_2d):
            self.annotations.append(coco_format_annotation(self.obj_count, self.img_count, labels[v_idx], vert)) 
            self.obj_count += 1
        
        with open(os.path.join(self.basepath, self.dataset_paths['labels'], 'coco_annotations.json'), 'w') as f:
            json_str = json.dumps({
                "info": self.info,
                "images": self.images,
                "annotations": self.annotations,
                "licenses": self.licenses,
                "categories": COCO_CATEGORIES()
            }, default=str)
            f.write(json_str)

        self.img_count += 1

    def _save_depth_ann(self, vert_2d, vert_cam, bboxes_in_image, labels, camera_params_list, dataset_split, orientations):
        filelist = ''.join(self.filelist)[:-1]
        with open(os.path.join(self.basepath, self.dataset_paths['filenames'], '%s_list.txt' % dataset_split), "w") as f:
            f.write(filelist)
        self.img_count += 1
    
        

    def save_camera_positions(self, info, camera_billboard_positions, camera_params_list):
        # if info['task'] != '3dod':
            # for key in camera_params_list[1].keys():
                # camera2_info[key] = None
        
        def camera_params_dict(camera_param):
            return_dict = {
                'height': camera_param.image_size[0],
                'width': camera_param.image_size[1], 
                'fov': camera_param.fov,
                'focal': camera_param.camera_focal
            }
            return return_dict

        with open(os.path.join(self.basepath, self.dataset_paths['labels'], 'camera_billboard_info.json'), 'w') as f:
            json_str = json.dumps({
                "info": info,
                "camera_billboard_positions": camera_billboard_positions,
                "patch_mask_corners": {},
                "camera_params": camera_params_dict(camera_params_list[0]),
                "camera_dx_params": camera_params_dict(camera_params_list[1])
            }, default=str)
            f.write(json_str)



# Returns specific data folder structure for each task
def set_dataset_paths(task, root_path, billboard_name, dataset_split):
    # Semantic Segmentation -> cityscapes
    if task == 'ss':
        dataset_paths = {
            'images': os.path.join(root_path, billboard_name, 'leftImg8bit', dataset_split, dataset_split),
            'labels': os.path.join(root_path, billboard_name, 'gtFine', dataset_split, dataset_split)
        }
        config_file = 'cityscapes.json'
    # 2d Object Detection -> COCO
    elif task == '2dod':
        dataset_paths = {
            'images': os.path.join(root_path, billboard_name, 'images', dataset_split),
            'labels': os.path.join(root_path, billboard_name, 'annotations', dataset_split)
        }
        config_file = 'coco.json'
    # 3D Object Detection -> Kitti
    elif task == '3dod':
        dataset_paths = {
            'images': os.path.join(root_path, billboard_name, dataset_split, 'image_2'),
            'images_dx': os.path.join(root_path, billboard_name, dataset_split, 'image_3'),
            'calib': os.path.join(root_path, billboard_name, dataset_split, 'calib'),
            'labels': os.path.join(root_path, billboard_name, dataset_split, 'label_2'),
            'splits': os.path.join(root_path, billboard_name, 'splits')
        }
        config_file = 'kitti.json'
    # Depth Estimation -> Kitti
    elif task == 'depth':
        dataset_paths = {
            'images': os.path.join(root_path, billboard_name, 'leftImg8bit', dataset_split),
            'labels': os.path.join(root_path, billboard_name, 'gtFine', dataset_split),
            'filenames': os.path.join(root_path, billboard_name, 'filenames'),
        }
        config_file = 'kitti.json'
    return dataset_paths, config_file


# write annotation dict in coco format
def coco_format_annotation(ann_id, image_id, category_id, vertices_2d, segmentation=[], is_crowd=0):
    x, y = int(vertices_2d[0, 0]), int(vertices_2d[0, 1])
    w, h = int(vertices_2d[-1, 0]) - x, int(vertices_2d[-1, 1]) - y

    annotation = {
        "id": ann_id, 
        "image_id": image_id, 
        "category_id": COCO_CLASSES[category_id], 
        "segmentation": segmentation, 
        "area": w * h, 
        "bbox": [x, y, w, h], 
        "iscrowd": is_crowd,
    }
    # categories[{
    #     "id": int, "name": str, "supercategory": str,
    #     }]
    return annotation

# write image dict in coco format
def coco_format_image(image_id, image_shape, file_name):
    
    image = {
            "id": image_id,
            "width": image_shape[1],
            "height": image_shape[0],
            "file_name": file_name,
            "license": 0,
            "flickr_url": '',
            "coco_url": '',
            "date_captured": datetime.datetime.now() 
            }
    # categories[{
    #     "id": int, "name": str, "supercategory": str,
    #     }]
    return image


def kitti_format_annotation(bbox, vertices_2d, v_cam, label, camera_transform, orientation):
    truncate = 0
    occluded = 0

    # Fix labels
    if not ('car' in label or 'pedestrian' in label):
        label = 'DontCare'
    else:
        label = label[0].upper() + label[1:]

    # 2d bbox
    x1, y1 = int(vertices_2d[0, 0]), int(vertices_2d[0, 1])
    x2, y2 = int(vertices_2d[-1, 0]), int(vertices_2d[-1, 1])

    # 3d bbox
    H, W, L = 2 * bbox.extent.z, 2 * bbox.extent.y, 2 * bbox.extent.x
    # print(v_cam)
    x_c, y_c, z_c = np.mean(v_cam[0, :]), np.mean(v_cam[1, :]), np.mean(v_cam[2, :])

    # orientation of the bbox wrt camera
    r_y = - np.pi/180 * orientation # bbox.rotation.yaw # (bbox.rotation.yaw - camera_transform.rotation.yaw)
    r_y = (r_y + np.pi) % (2 * np.pi) - np.pi # Mod(fAng + _PI, _TWO_PI) - _PI

    # alpha is the object viewing angle.
    dx = x_c - camera_transform.location.x
    dz = z_c - camera_transform.location.z
    alpha = np.arctan2(dz, dx)
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    return '%s %.2f %d %.2f %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (label, truncate, occluded, alpha, x1, y1, x2, y2, H, W, L, x_c, y_c, z_c, r_y)


def kitti_format_calib(camera_params2, camera_params3):
    P2 = np.block([camera_params2.intrinsics, np.zeros((3, 1))]) @ np.eye(4) #camera_params2.extrinsics
    P3 = np.block([camera_params3.intrinsics, np.zeros((3, 1))]) @ (camera_params3.extrinsics @ np.linalg.inv(camera_params2.extrinsics) )
    R0_rect = np.eye(3)
    Tr_imu_to_cam = np.block([np.eye(3), np.zeros((3, 1))])
    Tr_velo_to_cam = np.block([np.eye(3), np.zeros((3, 1))])

    # P0, P1, P2, P3
    calib_str = 'P0: ' + ' '.join(['%e' % P2[i, j] for i in range(3) for j in range(4)]) + '\n'
    calib_str += 'P1: ' + ' '.join(['%e' % P3[i, j] for i in range(3) for j in range(4)]) + '\n'
    calib_str += 'P2: ' + ' '.join(['%e' % P2[i, j] for i in range(3) for j in range(4)]) + '\n'
    calib_str += 'P3: ' + ' '.join(['%e' % P3[i, j] for i in range(3) for j in range(4)]) + '\n'
    calib_str += 'R0_rect: ' + ' '.join(['%e' % R0_rect[i, j] for i in range(3) for j in range(3)]) + '\n'
    calib_str += 'Tr_velo_to_cam: ' + ' '.join(['%e' % Tr_velo_to_cam[i, j] for i in range(3) for j in range(4)]) + '\n'
    calib_str += 'Tr_imu_to_velo: ' + ' '.join(['%e' % Tr_imu_to_cam[i, j] for i in range(3) for j in range(4)])

    return calib_str


def json_billboard_position(billboards_transform, cameras_transform, count):
    camera_transform, camera_transform_dx = cameras_transform
    billboard_data = [None] * 6
    billboard2_data = [None] * 6
    deg2rad = np.pi / 180

    num_billboards = len(billboards_transform)
    if num_billboards > 0:
        billboard_data = [billboards_transform[0].location.x, -billboards_transform[0].location.y, billboards_transform[0].location.z, 
                            deg2rad * billboards_transform[0].rotation.roll, deg2rad * billboards_transform[0].rotation.pitch, -deg2rad * billboards_transform[0].rotation.yaw]
    if num_billboards > 1:
        billboard2_data = [billboards_transform[1].location.x, -billboards_transform[1].location.y, billboards_transform[1].location.z, 
                            deg2rad * billboards_transform[1].rotation.roll, deg2rad * billboards_transform[1].rotation.pitch, -deg2rad * billboards_transform[1].rotation.yaw]

    # TODO include patch mask, or compute it in carla-loader.
    return_dict = {
        'id': count,
        'billboard_xyzrpy': billboard_data,
        'billboard2_xyzrpy': billboard2_data,
        'camera_xyzrpy': [camera_transform.location.x, -camera_transform.location.y, camera_transform.location.z, 
                            deg2rad * camera_transform.rotation.roll, deg2rad * camera_transform.rotation.pitch, -deg2rad * camera_transform.rotation.yaw],
        'camera_dx_xyzrpy': [camera_transform_dx.location.x, -camera_transform_dx.location.y, camera_transform_dx.location.z, 
                            deg2rad * camera_transform_dx.rotation.roll, deg2rad * camera_transform_dx.rotation.pitch, -deg2rad * camera_transform_dx.rotation.yaw]
    }
    return return_dict

def COCO_CATEGORIES():
    return [{"supercategory": "person","id": 1,"name": "person"},
            {"supercategory": "vehicle","id": 2,"name": "bicycle"},
            {"supercategory": "vehicle","id": 3,"name": "car"},
            {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
            {"supercategory": "vehicle","id": 5,"name": "airplane"},
            {"supercategory": "vehicle","id": 6,"name": "bus"},
            {"supercategory": "vehicle","id": 7,"name": "train"},
            {"supercategory": "vehicle","id": 8,"name": "truck"},
            {"supercategory": "vehicle","id": 9,"name": "boat"},
            {"supercategory": "outdoor","id": 10,"name": "traffic light"},
            {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
            {"supercategory": "outdoor","id": 13,"name": "stop sign"},
            {"supercategory": "outdoor","id": 14,"name": "parking meter"},
            {"supercategory": "outdoor","id": 15,"name": "bench"},
            {"supercategory": "animal","id": 16,"name": "bird"},
            {"supercategory": "animal","id": 17,"name": "cat"},
            {"supercategory": "animal","id": 18,"name": "dog"},
            {"supercategory": "animal","id": 19,"name": "horse"},
            {"supercategory": "animal","id": 20,"name": "sheep"},
            {"supercategory": "animal","id": 21,"name": "cow"},
            {"supercategory": "animal","id": 22,"name": "elephant"},
            {"supercategory": "animal","id": 23,"name": "bear"},
            {"supercategory": "animal","id": 24,"name": "zebra"},
            {"supercategory": "animal","id": 25,"name": "giraffe"},
            {"supercategory": "accessory","id": 27,"name": "backpack"},
            {"supercategory": "accessory","id": 28,"name": "umbrella"},
            {"supercategory": "accessory","id": 31,"name": "handbag"},
            {"supercategory": "accessory","id": 32,"name": "tie"},
            {"supercategory": "accessory","id": 33,"name": "suitcase"},
            {"supercategory": "sports","id": 34,"name": "frisbee"},
            {"supercategory": "sports","id": 35,"name": "skis"},
            {"supercategory": "sports","id": 36,"name": "snowboard"},
            {"supercategory": "sports","id": 37,"name": "sports ball"},
            {"supercategory": "sports","id": 38,"name": "kite"},
            {"supercategory": "sports","id": 39,"name": "baseball bat"},
            {"supercategory": "sports","id": 40,"name": "baseball glove"},
            {"supercategory": "sports","id": 41,"name": "skateboard"},
            {"supercategory": "sports","id": 42,"name": "surfboard"},
            {"supercategory": "sports","id": 43,"name": "tennis racket"},
            {"supercategory": "kitchen","id": 44,"name": "bottle"},
            {"supercategory": "kitchen","id": 46,"name": "wine glass"},
            {"supercategory": "kitchen","id": 47,"name": "cup"},
            {"supercategory": "kitchen","id": 48,"name": "fork"},
            {"supercategory": "kitchen","id": 49,"name": "knife"},
            {"supercategory": "kitchen","id": 50,"name": "spoon"},
            {"supercategory": "kitchen","id": 51,"name": "bowl"},
            {"supercategory": "food","id": 52,"name": "banana"},
            {"supercategory": "food","id": 53,"name": "apple"},
            {"supercategory": "food","id": 54,"name": "sandwich"},
            {"supercategory": "food","id": 55,"name": "orange"},
            {"supercategory": "food","id": 56,"name": "broccoli"},
            {"supercategory": "food","id": 57,"name": "carrot"},
            {"supercategory": "food","id": 58,"name": "hot dog"},
            {"supercategory": "food","id": 59,"name": "pizza"},
            {"supercategory": "food","id": 60,"name": "donut"},
            {"supercategory": "food","id": 61,"name": "cake"},
            {"supercategory": "furniture","id": 62,"name": "chair"},
            {"supercategory": "furniture","id": 63,"name": "couch"},
            {"supercategory": "furniture","id": 64,"name": "potted plant"},
            {"supercategory": "furniture","id": 65,"name": "bed"},
            {"supercategory": "furniture","id": 67,"name": "dining table"},
            {"supercategory": "furniture","id": 70,"name": "toilet"},
            {"supercategory": "electronic","id": 72,"name": "tv"},
            {"supercategory": "electronic","id": 73,"name": "laptop"},
            {"supercategory": "electronic","id": 74,"name": "mouse"},
            {"supercategory": "electronic","id": 75,"name": "remote"},
            {"supercategory": "electronic","id": 76,"name": "keyboard"},
            {"supercategory": "electronic","id": 77,"name": "cell phone"},
            {"supercategory": "appliance","id": 78,"name": "microwave"},
            {"supercategory": "appliance","id": 79,"name": "oven"},
            {"supercategory": "appliance","id": 80,"name": "toaster"},
            {"supercategory": "appliance","id": 81,"name": "sink"},
            {"supercategory": "appliance","id": 82,"name": "refrigerator"},
            {"supercategory": "indoor","id": 84,"name": "book"},
            {"supercategory": "indoor","id": 85,"name": "clock"},
            {"supercategory": "indoor","id": 86,"name": "vase"},
            {"supercategory": "indoor","id": 87,"name": "scissors"},
            {"supercategory": "indoor","id": 88,"name": "teddy bear"},
            {"supercategory": "indoor","id": 89,"name": "hair drier"},
            {"supercategory": "indoor","id": 90,"name": "toothbrush"}]


COCO_CLASSES = {
    'pedestrian': 1,
    'car': 3,
    'bicycle': 2,
    'motorcycle': 4,
    'truck': 8,
    'stop_sign': 13,
    'traffic_light': 10
}

