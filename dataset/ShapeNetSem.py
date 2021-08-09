from dataset_utils import Dataset
from glob import glob
import os

from ModelNet import ModelNet40PLY, CLASS_DICT_40


SHAPENETSEM_DIR = "/home/jbweibel/dataset/ShapeNetSem/"
SCANNET_DIR = '/home/jbweibel/dataset/ScanNet/'


# CLASS_DICT = {
#     # 'bathtub': 0,
#     'bed': 0,
#     'bookshelf': 1,
#     'cabinet': 2,
#     'chair': 3,
#     ## 'counter': 5,
#     # 'curtain': 5,
#     'desk': 4,
#     'door': 5,
#     ## 'fridge': 9,
#     ## 'otherfurnitures': ,
#     ## 'picture': 10,
#     ## 'shower curtain': ,
#     'sink': 6,
#     'sofa': 7,
#     'table': 8,
#     'toilet': 9,
#     ## 'window': 15,
# }

CLASS_DICT = {
    'bed': 0,
    'cabinet': 1,
    'chair': 2,
    'desk': 3,
    'display': 4,
    'door': 5,
    'shelf': 6,
    'sink': 7,
    'sofa': 8,
    'table': 9,
    'toilet': 10,
}



MODELNET_TO_SHAPENET = {
    1: "bathtub",
    2: "bed",
    4: "bookshelf",
    8: "chair",
    11: "curtain",
    12: "desk",
    13: "door",
    14: "cabinet",
    # 22: "display",
    3: "chair",
    29: "sink",
    30: "sofa",
    32: "chair",
    33: "table",
    35: "toilet",
    38: "cabinet",
}


NYU40_ID_TO_LABEL = {
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "shelf",
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


class ShapeNetSemToScanNetDet(Dataset):
    def __init__(self, **kwargs):
        super(ShapeNetSemToScanNetDet, self).__init__(**kwargs)
        global CLASS_DICT
        self.class_dict = CLASS_DICT
        self.modelnet = ModelNet40PLY(**kwargs)

        with open(SCANNET_DIR + "scannetv2_val.txt") as fp:
            val_scenes = fp.readlines()
        self.val_scenes = [scene.strip() for scene in val_scenes]

    def nyu_to_sn(self, nyu_id):
        label = NYU40_ID_TO_LABEL[nyu_id]
        if label in self.class_dict:
            return self.class_dict[NYU40_ID_TO_LABEL[nyu_id]]
        else:
            return -1

    def get_train_test_dataset(self):
        # Obtain train set from ModelNet
        train_set = [[] for i in range(len(self.class_dict))]
        # mn_train_set, _ = self.modelnet.get_train_test_dataset()
        # for mn_cls_idx, mn_cls_label in MODELNET_TO_SHAPENET.items():
        #     if not mn_cls_label in self.class_dict:
        #         continue
        #     dataset_idx = self.class_dict[mn_cls_label]
        #     train_set[dataset_idx] += mn_train_set[mn_cls_idx]

        # Complete train set using ShapeNetSem
        for cls_label in self.class_dict.keys():
            # if not cls_label in self.class_dict:
            #     continue
            dataset_idx = self.class_dict[cls_label]
            train_set[dataset_idx] += glob(os.path.join(
                SHAPENETSEM_DIR, cls_label, "*"))

        # Test set from ScanNetDetection
        test_set = [["{}scans/{}/{}_filtered.ply".format(
            SCANNET_DIR, scan, scan) for scan in self.val_scenes]]

        return train_set, test_set
