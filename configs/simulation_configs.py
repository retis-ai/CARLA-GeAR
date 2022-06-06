import numpy as np
import os

# Seeds for repeatability
SEED_VALUES = {
    'finetuning': 0,
    'train': 123,
    'val': 131,
    'test': 363,  # 363
    'debug': 10,
    #'finetuningval': 100,
    #'finetuningtest': 1000,
    'video': 60,  #Good values: 313
}

ITERATION_VALUES = {
    'finetuning': 300, 
    'train': 100,
    'val': 50,
    'test': 50, 
    'debug': 2, 
    #'finetuningval': 50,  ##25,
    #'finetuningtest': 50,  ##25,
    'video': 1,
}

# Vehicle brands included in the simulation. Do not include ford or audi.
BRANDS = ['dodge', 'citroen', 'chevrolet', 'bmw', 'toyota', 'mercedes']
MAX_VEHICLE_NPC = 7
MAX_WALKER_NPC = 200

def get_spawn_bp(billboard):
    if 'no_billboard' in billboard:
        return None
    elif 'billboard' in billboard:
        return 'vehicle.audi.a2'
    elif 'truck' in billboard:
        return 'vehicle.ford.ambulance'
    else:
        raise NotImplemented('No billboard name %s is implemented.' % billboard)

# CarlaUE4 import path for patch.
patch_load_path = '/home/federico/carla/carla/Unreal/CarlaUE4/Content/patches/patch_auto/patch.png'




    
