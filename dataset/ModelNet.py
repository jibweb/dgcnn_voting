from dataset_utils import Dataset
from functools import partial
from glob import glob
import pickle

# Train / Test directories
DATA_DIR = "/home/jbweibel/dataset/ModelNet/"
TRAIN_DIR_40 = DATA_DIR + "modelnet40_manually_aligned_TrainPc/"
TEST_DIR_40 = DATA_DIR + "modelnet40_manually_aligned_TestPc/"

TRAIN_DIR_10 = DATA_DIR + "ModelNet10_TrainPc/"
TEST_DIR_10 = DATA_DIR + "ModelNet10_TestPc/"

TRAIN_PLY_DIR_10 = DATA_DIR + "ModelNet10_TrainPly/"
TEST_PLY_DIR_10 = DATA_DIR + "ModelNet10_TestPly/"
TRAIN_PLY_DIR_40 = DATA_DIR + "modelnet40_manually_aligned_TrainPly/"
TEST_PLY_DIR_40 = DATA_DIR + "modelnet40_manually_aligned_TestPly/"

OFF_DIR_10 = DATA_DIR + "ModelNet10/"
OFF_DIR_40 = DATA_DIR + "modelnet40_manually_aligned/"


# SN_DATA_DIR = '/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/ScanNet/'
SN_DATA_DIR = '/home/jbweibel/dataset/ScanNet/'

CLASS_DICT_40 = {
    "airplane": 0,
    "bathtub": 1,
    "bed": 2,
    "bench": 3,
    "bookshelf": 4,
    "bottle": 5,
    "bowl": 6,
    "car": 7,
    "chair": 8,
    "cone": 9,
    "cup": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "dresser": 14,
    "flower_pot": 15,
    "glass_box": 16,
    "guitar": 17,
    "keyboard": 18,
    "lamp": 19,
    "laptop": 20,
    "mantel": 21,
    "monitor": 22,
    "night_stand": 23,
    "person": 24,
    "piano": 25,
    "plant": 26,
    "radio": 27,
    "range_hood": 28,
    "sink": 29,
    "sofa": 30,
    "stairs": 31,
    "stool": 32,
    "table": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39,
}

CLASS_DICT_10 = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'desk': 3,
    'dresser': 4,
    'monitor': 5,
    'night_stand': 6,
    'sofa': 7,
    'table': 8,
    'toilet': 9,
}


class _ModelNet(Dataset):
    def __init__(self,
                 class_dict=CLASS_DICT_40,
                 train_dir="/train",
                 test_dir="/test",
                 train_regex="/*regex.pcd",
                 test_regex="/*regex.pcd",
                 **kwargs):
        super(_ModelNet, self).__init__(**kwargs)
        self.class_dict = class_dict
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_regex = train_regex
        self.test_regex = test_regex

    def get_train_test_dataset(self):
        train_set = [glob(self.train_dir + cls + self.train_regex)
                     for cls in sorted(self.class_dict.keys())]
        test_set = [glob(self.test_dir + cls + self.test_regex)
                    for cls in sorted(self.class_dict.keys())]

        return train_set, test_set


# === MODELNET40 DATASETS =====================================================
ModelNet40PCD = partial(_ModelNet, class_dict=CLASS_DICT_40,
                        train_dir=TRAIN_DIR_40, test_dir=TEST_DIR_40,
                        train_regex="/*_full_wnormals_wattention.pcd",
                        test_regex="/*_full_wnormals_wattention.pcd")

ModelNet40OFF = partial(_ModelNet, class_dict=CLASS_DICT_40,
                        train_dir=OFF_DIR_40, test_dir=OFF_DIR_40,
                        train_regex="/train/*.off",
                        test_regex="/test/*.off")

ModelNet40PLY = partial(_ModelNet, class_dict=CLASS_DICT_40,
                        train_dir=TRAIN_PLY_DIR_40, test_dir=TEST_PLY_DIR_40,
                        train_regex="/*_visible_normals_bin.ply",
                        test_regex="/*_visible_normals_bin.ply")


# === MODELNET10 DATASETS =====================================================
ModelNet10PCD = partial(_ModelNet, class_dict=CLASS_DICT_10,
                        train_dir=TRAIN_DIR_10, test_dir=TEST_DIR_10,
                        train_regex="/*_full_wnormals_wattention.pcd",
                        test_regex="/*_full_wnormals_wattention.pcd")

ModelNet10OFF = partial(_ModelNet, class_dict=CLASS_DICT_10,
                        train_dir=OFF_DIR_10, test_dir=OFF_DIR_10,
                        train_regex="/train/*.off",
                        test_regex="/test/*.off")

ModelNet10PLY = partial(_ModelNet, class_dict=CLASS_DICT_10,
                        train_dir=TRAIN_PLY_DIR_10, test_dir=TEST_PLY_DIR_10,
                        train_regex="/*[0-9]_visible_normals_bin.ply",
                        test_regex="/*[0-9]_visible_normals_bin.ply")


class _ScanNetToModelNet(Dataset):
    def __init__(self,
                 class_dict=CLASS_DICT_40,
                 sn_to_mn_fn="",
                 mn_to_sn_fn="",
                 regex="*_wnormals.ply",
                 **kwargs):

        # TODO: fix balance_by_oversampling when class has 0 elements
        # Forcing it because some class don't have any elements !
        kwargs["balance_train_set"] = False
        super(_ScanNetToModelNet, self).__init__(**kwargs)

        self.class_dict = class_dict

        global SN_DATA_DIR
        self.regex = regex

        with open(SN_DATA_DIR + "scannetv1_test.txt") as fp:
            test_scenes = fp.readlines()
        self.test_scenes = [scene.strip() for scene in test_scenes]

        with open(SN_DATA_DIR + "scannetv1_train.txt") as fp:
            train_scenes = fp.readlines()
        self.train_scenes = [scene.strip() for scene in train_scenes]

        with open(SN_DATA_DIR + "scannetv1_val.txt") as fp:
            val_scenes = fp.readlines()
        self.val_scenes = [scene.strip() for scene in val_scenes]

        with open(SN_DATA_DIR + sn_to_mn_fn, "rb") as fp:
            self.map_SN_MN = pickle.load(fp)

        with open(SN_DATA_DIR + mn_to_sn_fn, "rb") as fp:
            self.map_MN_SN = pickle.load(fp)

    def get_set_from_scenes(self, scenes):
        object_set = []
        for scene in scenes:
            object_set += [fn for fn in
                           glob("{}scans/{}/objectsv2/*{}".format(SN_DATA_DIR,
                                                                  scene,
                                                                  self.regex))
                           if fn.split("/")[-1].split("_")[0] in self.map_SN_MN]

        return [[fn for fn in object_set
                 if self.map_SN_MN[fn.split("/")[-1].split("_")[0]] == cls]
                for cls in sorted(self.class_dict.keys())]

    def get_train_test_dataset(self):
        train_set = self.get_set_from_scenes(self.train_scenes)
        test_set = self.get_set_from_scenes(self.test_scenes)
        val_set = self.get_set_from_scenes(self.val_scenes)

        return train_set, test_set, val_set


ScanNetToModelNet10 = partial(_ScanNetToModelNet, class_dict=CLASS_DICT_10,
                              sn_to_mn_fn="map_SN_MN10.pickle",
                              mn_to_sn_fn="map_MN10_SN.pickle",
                              regex="*_wnormals.ply")

ScanNetToModelNet40 = partial(_ScanNetToModelNet, class_dict=CLASS_DICT_40,
                              sn_to_mn_fn="map_SN_MN40.pickle",
                              mn_to_sn_fn="map_MN40_SN.pickle",
                              regex="*_wnormals.ply")
