from dataset_utils import Dataset
from functools import partial
from glob import glob
import os

from ModelNet import ModelNet40PLY, ScanNetToModelNet40

DATA_DIR = "/home/jbweibel/dataset/ScanObjectNN/"

# Classes in alphabetical order, as opposed to original order
CLASS_DICT = {
    "bag": 0,
    "bed": 1,
    "bin": 2,
    "box": 3,
    "cabinet": 4,
    "chair": 5,
    "desk": 6,
    "display": 7,
    "door": 8,
    "pillow": 9,
    "shelf": 10,
    "sink": 11,
    "sofa": 12,
    "table": 13,
    "toilet": 14,
}

CLASSES_ORIGINAL_ORDER = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]

# ScanObjectNN classes present in ModelNet, reordered from 0
OBJECTDATASET_TO_COMBINED = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    12: 8,
    13: 9,
    14: 10,
}

# ModelNet index to index in CLASSES_ORIGINAL_ORDER
MODELNET_TO_OBJECTDATASET = {
    2: 10,
    4: 8,
    8: 4,
    12: 5,
    13: 7,
    14: 3,
    22: 6,
    3: 4,
    29: 12,
    30: 13,
    32: 4,
    33: 9,
    35: 14,
    38: 3
}

CLASS_DICT_COMBINED = {
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


class _ScanObjectNN(Dataset):
    def __init__(self,
                 base_dir="ScanObjectNNvariant",
                 **kwargs):
        super(_ScanObjectNN, self).__init__(**kwargs)
        self.class_dict = CLASS_DICT
        self.base_dir = base_dir

        # Classes defined in the order they were defined in the original dataset
        self.classes = CLASSES_ORIGINAL_ORDER

    def get_train_test_dataset(self):
        with open(DATA_DIR + "object_dataset/main_split.txt") as fp:
            splits = [line.strip().split("\t") for line in fp.readlines()]

        # classes = sorted(self.class_dict.keys())
        train_set = [[] for i in range(len(self.class_dict))]
        test_set = [[] for i in range(len(self.class_dict))]

        for fn in splits:
            cls = self.classes[int(fn[1])]
            formatted_fn = "{}{}/{}/{}".format(DATA_DIR,
                                               self.base_dir,
                                               cls, fn[0])
            if len(fn) == 2:
                if self.base_dir == "object_dataset":
                    train_set[self.class_dict[cls]].append(formatted_fn)
                else:
                    train_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_0.bin")
                    train_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_1.bin")
                    train_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_2.bin")
                    train_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_3.bin")
                    train_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_4.bin")
            elif len(fn) == 3:
                if self.base_dir == "object_dataset":
                    test_set[self.class_dict[cls]].append(formatted_fn)
                else:
                    test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_0.bin")
                    test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_1.bin")
                    test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_2.bin")
                    test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_3.bin")
                    test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_4.bin")
        return train_set, test_set


ScanObjectNN = partial(_ScanObjectNN, base_dir="object_dataset")
ScanObjectNN_T25 = partial(_ScanObjectNN, base_dir="PB_T25")
ScanObjectNN_T50_RS = partial(_ScanObjectNN, base_dir="PB_T50_RS")


class _ScanObjectNNToModelNet(Dataset):
    def __init__(self,
                 base_dir="ScanObjectNNvariant",
                 **kwargs):
        super(_ScanObjectNNToModelNet, self).__init__(**kwargs)

        self.class_dict = CLASS_DICT_COMBINED
        self.classes = CLASSES_ORIGINAL_ORDER
        self.base_dir = base_dir
        self.modelnet = ModelNet40PLY(**kwargs)

    def get_train_test_dataset(self):
        # Obtain train set from ModelNet
        train_set = [[] for i in range(len(self.class_dict))]
        mn_train_set, _ = self.modelnet.get_train_test_dataset()
        for mn_cls_idx, sonn_cls_idx in MODELNET_TO_OBJECTDATASET.items():
            dataset_idx = self.class_dict[self.classes[sonn_cls_idx]]
            train_set[dataset_idx] += mn_train_set[mn_cls_idx]

        # Obtain test set from ScanObjectNN
        with open(DATA_DIR + "object_dataset/main_split.txt") as fp:
            splits = [line.strip().split("\t") for line in fp.readlines()]

        test_set = [[] for i in range(len(self.class_dict))]
        test_elts = [fn for fn in splits if
                         len(fn) == 3 and
                         int(fn[1]) in OBJECTDATASET_TO_COMBINED]

        for fn in test_elts:
            cls = self.classes[int(fn[1])]
            formatted_fn = os.path.join(DATA_DIR,
                                        self.base_dir,
                                        cls,
                                        fn[0])
            if self.base_dir == "object_dataset":
                test_set[self.class_dict[cls]].append(formatted_fn)
            else:
                test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_0.bin")
                test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_1.bin")
                test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_2.bin")
                test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_3.bin")
                test_set[self.class_dict[cls]].append(formatted_fn[:-4] + "_4.bin")


        return train_set, test_set


ScanObjectNNToModelNet = partial(_ScanObjectNNToModelNet, base_dir="object_dataset")
ScanObjectNN_T25ToModelNet = partial(_ScanObjectNNToModelNet, base_dir="PB_T25")
ScanObjectNN_T50_RSToModelNet = partial(_ScanObjectNNToModelNet, base_dir="PB_T50_RS")


# class ScanObjectNNToSONN11(Dataset):
#     def __init__(self,
#                  **kwargs):
#         super(ScanObjectNNToSONN11, self).__init__(**kwargs)

#         self.class_dict = CLASS_DICT_COMBINED
#         self.classes = CLASSES_ORIGINAL_ORDER

#     def get_train_test_dataset(self):
#         with open(DATA_DIR + "object_dataset/split_new.txt") as fp:
#             splits = [line.strip( ).split("\t") for line in fp.readlines()]

#         train_set = [[] for i in range(len(self.class_dict))]
#         test_set = [[] for i in range(len(self.class_dict))]

#         for fn in splits:
#             if not int(fn[1]) in OBJECTDATASET_TO_COMBINED:
#                 continue

#             cls = self.classes[int(fn[1])]
#             formatted_fn = os.path.join(DATA_DIR,
#                                         "object_dataset",
#                                         cls,
#                                         fn[0])
#             if len(fn) == 2:
#                 train_set[self.class_dict[cls]].append(formatted_fn)
#             elif len(fn) == 3:
#                 test_set[self.class_dict[cls]].append(formatted_fn)

#         return train_set, test_set


# class ModelNetToSONN11(Dataset):
#     def __init__(self,
#                  **kwargs):
#         super(ModelNetToSONN11, self).__init__(**kwargs)

#         self.class_dict = CLASS_DICT_COMBINED
#         self.classes = CLASSES_ORIGINAL_ORDER
#         self.modelnet = ModelNet40PLY(**kwargs)

#     def get_train_test_dataset(self):
#         # Obtain train set from ModelNet
#         train_set = [[] for i in range(len(self.class_dict))]
#         test_set = [[] for i in range(len(self.class_dict))]

#         mn_train_set, mn_test_set = self.modelnet.get_train_test_dataset()
#         for mn_cls_idx, sonn_cls_idx in MODELNET_TO_OBJECTDATASET.items():
#             dataset_idx = self.class_dict[self.classes[sonn_cls_idx]]
#             train_set[dataset_idx] += mn_train_set[mn_cls_idx]
#             test_set[dataset_idx] += mn_test_set[mn_cls_idx]

#         return train_set, test_set


# class ScanNetToSONN11(Dataset):
#     def __init__(self,
#                  **kwargs):
#         super(ScanNetToSONN11, self).__init__(**kwargs)

#         self.class_dict = CLASS_DICT_COMBINED
#         self.classes = CLASSES_ORIGINAL_ORDER
#         self.scannet = ScanNetToModelNet40(**kwargs)

#     def get_train_test_dataset(self):
#         # Obtain train set from ScanNet
#         train_set = [[] for i in range(len(self.class_dict))]
#         test_set = [[] for i in range(len(self.class_dict))]
#         val_set = [[] for i in range(len(self.class_dict))]

#         sn_train_set, sn_test_set, sn_val_set = self.scannet.get_train_test_dataset()
#         for mn_cls_idx, sonn_cls_idx in MODELNET_TO_OBJECTDATASET.items():
#             dataset_idx = self.class_dict[self.classes[sonn_cls_idx]]
#             train_set[dataset_idx] += sn_train_set[mn_cls_idx]
#             test_set[dataset_idx] += sn_test_set[mn_cls_idx]
#             val_set[dataset_idx] += sn_val_set[mn_cls_idx]

#         return train_set, test_set, val_set
