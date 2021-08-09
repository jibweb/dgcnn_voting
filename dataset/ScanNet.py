from dataset_utils import Dataset
from glob import glob

DATA_DIR = '/home/jbweibel/dataset/ScanNet/'

# CLASS_DICT = {
#     'basket': 0,
#     'bathtub': 1,
#     'bed': 2,
#     'cabinet': 3,
#     'chair': 4,
#     'keyboard': 5,
#     'lamp': 6,
#     'laptop': 7,
#     'microwave': 8,
#     'pillow': 9,
#     'printer': 10,
#     'shelf': 11,
#     'sofa': 12,
#     'stove': 13,
#     'table': 14,
#     'trash': 15,
#     'tv': 16,
# }

CLASS_DICT = {
    'basket': 0,
    'bathtub': 1,
    'bed': 2,
    'cabinet': 3,
    'chair': 4,
    'keyboard': 5,
    'lamp': 6,
    'microwave': 7,
    'pillow': 8,
    'printer': 9,
    'shelf': 10,
    'stove': 11,
    'table': 12,
    'tv': 13,
}


class ScanNet(Dataset):
    def __init__(self,
                 regex="*_wnormals.ply",
                 **kwargs):
        super(ScanNet, self).__init__(**kwargs)
        global CLASS_DICT, DATA_DIR
        self.class_dict = CLASS_DICT
        self.regex = regex

        with open(DATA_DIR + "scannetv1_test.txt") as fp:
            test_scenes = fp.readlines()
        self.test_scenes = [scene.strip() for scene in test_scenes]

        with open(DATA_DIR + "scannetv1_train.txt") as fp:
            train_scenes = fp.readlines()
        self.train_scenes = [scene.strip() for scene in train_scenes]

        with open(DATA_DIR + "scannetv1_val.txt") as fp:
            val_scenes = fp.readlines()
        self.val_scenes = [scene.strip() for scene in val_scenes]

    def get_set_from_scenes(self, scenes):
        dataset = []
        for cls in sorted(self.class_dict):
            cls_set = []
            for scene in scenes:
                cls_set += glob("{}scans/{}/objectsv2/{}{}".format(DATA_DIR,
                                                                   scene,
                                                                   cls,
                                                                   self.regex))
            dataset.append(cls_set)
        return dataset

    def get_train_test_dataset(self):
        train_set = self.get_set_from_scenes(self.train_scenes)
        test_set = self.get_set_from_scenes(self.test_scenes)
        val_set = self.get_set_from_scenes(self.val_scenes)

        return train_set, test_set, val_set


NYU40_ID_TO_LABEL = {
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "fridge",
    28: "shower_curtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
    39: "otherfurnitures",
}

DET_CLASS_DICT = {
    "bathtub": 0,
    "bed": 1,
    "bookshelf": 2,
    "cabinet": 3,
    "chair": 4,
    "counter": 5,
    "curtain": 6,
    "desk": 7,
    "door": 8,
    "fridge": 9,
    "otherfurnitures": 10,
    "picture": 11,
    "shower_curtain": 12,
    "sink": 13,
    "sofa": 14,
    "table": 15,
    "toilet": 16,
    "window": 17,
}


class ScanNetDetection(Dataset):
    def __init__(self, **kwargs):
        super(ScanNetDetection, self).__init__(**kwargs)
        global DET_CLASS_DICT, NYU40_ID_TO_LABEL
        self.class_dict = DET_CLASS_DICT

        with open(DATA_DIR + "scannetv2_val.txt") as fp:
            val_scenes = fp.readlines()
        self.val_scenes = [scene.strip() for scene in val_scenes]

    def nyu_to_sn(self, nyu_id):
        return self.class_dict[NYU40_ID_TO_LABEL[nyu_id]]

    def get_train_test_dataset(self):
        train_set = [[]]
        val_set = [["{}scans/{}/{}_filtered.ply".format(DATA_DIR, scan, scan)
            for scan in self.val_scenes]]
        return train_set, val_set
