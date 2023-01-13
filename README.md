# CARLA-GeAR
This is the code repository for the data generation of our paper "[CARLA-GeAR: A Dataset Generator for Systematic Adversarial Robustness Evaluation of Vision Models](https://arxiv.org/abs/2206.04365)". More info can be found at the main project page: https://carlagear.retis.santannapisa.it/.

This code requires a working installation of CARLA to run. If you cannot/don't want to install the CARLA simulator, please consider using the datasets we provided in the main page.

![image](https://user-images.githubusercontent.com/92364988/172151551-640ca8ec-6159-4f4d-9c96-62ba58f4248e.png)

### Citation
If you found our work useful, please consider citing the paper!
```
@ARTICLE{2022arXiv220604365N,
       author = {{Nesti}, Federico and {Rossolini}, Giulio and {D'Amico}, Gianluca and {Biondi}, Alessandro and {Buttazzo}, Giorgio},
        title = "{CARLA-GeAR: a Dataset Generator for a Systematic Evaluation of Adversarial Robustness of Vision Models}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = jun,
          eid = {arXiv:2206.04365},
        pages = {arXiv:2206.04365},
archivePrefix = {arXiv},
       eprint = {2206.04365},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220604365N%7D,
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

### Features
This code allows dataset generation for a systematic evaluation of the adversarial robustness of custom models for 4 different tasks: **Semantic Segmentation** (SS), **2D object detection** (2DOD), **3D Stereo-camera Object Detection** (3DOD), and **Monocular Depth Estimation** (depth). The use of CARLA simulator allows photo-realistic rendering of the meshes, and full control of autonomous driving environment. Hence, it is possible to build several datasets that include surfaces on which an adversary might attach physically-realizable adversarial patches to change the network prediction.

The adversarial surfaces come in two different "shapes": billboards and a truck. The billboards are placed in fixed (customizable) positions in the selected city (suggested Town10HD), while the truck is spawned in random positions, always in front of the ego vehicle. It is possible to import the patches in CARLA, so that they are rendered realistically. 



### Installation
The code requires a running installation of CARLA. Since different versions might cause unexpected errors, we suggest to use the same setup we selected:
- Ubuntu 18.04
- CARLA 0.9.13
- Python3.6
Please refer to the step-by-step CARLA [installation guide](https://carla.readthedocs.io/en/latest/build_linux/) to install it smoothly.
After a successful installation of CARLA, create a virtualenv and install the Python API following the instructions.
You can then clone the repo in the parent folder where carla is installed.
Then, install the requirements: `pip install -r requirements.txt` using the same venv.

To allow easier billboard spawning and management we had to change a few of the original blueprints of CARLA. In particular, you will have to replace the `vehicle.audi.a2` (`carla/Unreal/CarlaUE4/Content/Carla/Blueprints/Vehicles/AudiA2/BP_AudiA2.uasset`) and `vehicle.ford.ambulance` (`carla/Unreal/CarlaUE4/Content/Carla/Blueprints/Vehicles/Ambulance/BP_Ambulance.uasset`) with the corresponding ones we provide in the `assets` folder.
Also, copy the files `assets/BP_Decal_Billboard_Auto.uasset` and `assets/Mat_patch.uasset` in `carla/Unreal/CarlaUE4/Content` and create a new folder `carla/Unreal/CarlaUE4/Content/patches/patch_auto/`.

### Running the code
To run the generation code, you will need to open the simulator (CarlaUE4.uproject), and run the simulation.
After this you can run the script that will collect a split of a dataset. 
`python collect_dataset.py --config <path_to_config_file>`
The config file is a `.yml` file (we provide examples for each task in the `config` folder), where it's possible to specify the task, the dataset save path, the billboard that must be placed in the city, and the patch that must be loaded (if any).

### Dataset format
Each task follows its own data format. For a smoother integration with already-working code, we decided to use the same data-format and folder structure of well-estabilished benchmarks. SS follows the CityScapes dataset format, 2DOD follows the COCO format, 3DOD and depth follow the corresponding Kitti format.

The saved dataset includes additional information on the cameras used, on the position of the camera and the billboards at each frame. This information might be used to perform digital patch placement during the optimization of patch-based attacks. Please check Patch Generation for further details on this.

### Billboard positioning
The dataset generation tool can be used with different objectives:
- Generic, fine-tuning datasets. This can be done by choosing the "no_billboard" option in the "billboard" field of the config file. The car will be spawned randomly in the city.
- Attack datasets.  

In the latter case, there are a few options. If you want to generate a dataset for a specific billboard, select "billboardXX". This will use the spawn info written in the json attack scenario library file. Otherwise, the "truck" option is available. In this scenario, the car is placed behind an "adversarial truck" with a patch on its rear end. 

Please note that in the yml you can select any city you want. However, since Town10HD is the most realistic one, we only provided pre-computed billboards positions only for that city. Different cities have not-so-realistic meshes that will make the networks perform poorly.

### Dataset Evaluation and Patch Generation
This repo includes only the data generation tool. If you want to perform patch-based attacks, or evaluate the performance of a network or a defense on any of these datasets, please check our tool at this repo: https://github.com/retis-ai/PatchAttackTool.







