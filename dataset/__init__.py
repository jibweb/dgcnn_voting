from enum import Enum
from ModelNet import ModelNet10PCD, CLASS_DICT_10 as MN10_CLASS_DICT
from ModelNet import ModelNet40PCD, CLASS_DICT_40 as MN40_CLASS_DICT
from ModelNet import ModelNet10OFF, ModelNet10PLY, ModelNet40OFF, ModelNet40PLY
from ModelNet import ScanNetToModelNet10, ScanNetToModelNet40
from ScanNet import ScanNet, CLASS_DICT as SN_CLASS_DICT
# from ScanNet import ScanNetDetection, DET_CLASS_DICT as SN_DET_CLASS_DICT
from ScanObjectNN import ScanObjectNN, ScanObjectNN_T25, ScanObjectNN_T50_RS, CLASS_DICT as SONN_CLASS_DICT
from ScanObjectNN import ScanObjectNNToModelNet, ScanObjectNN_T25ToModelNet, ScanObjectNN_T50_RSToModelNet, CLASS_DICT_COMBINED as SONN_CLASS_DICT_COMBINED
from ShapeNetSem import ShapeNetSemToScanNetDet, CLASS_DICT as SNS_CLASS_DICT


DATASETS = Enum("DATASET", ["ModelNet10PCD",
                            "ModelNet10OFF",
                            "ModelNet10PLY",

                            "ModelNet40PCD",
                            "ModelNet40OFF",
                            "ModelNet40PLY",

                            "ScanNet",
                            "ScanNetToModelNet10",
                            "ScanNetToModelNet40",

                            "ScanObjectNN",
                            "ScanObjectNN_T25",
                            "ScanObjectNN_T50_RS",
                            "ScanObjectNNToModelNet",
                            "ScanObjectNN_T25ToModelNet",
                            "ScanObjectNN_T50_RSToModelNet",
                            "ShapeNetSemToScanNetDet",
                            ])


def get_dataset(dataset_name):
    if type(dataset_name) == str:
        try:
            dataset_name = DATASETS[dataset_name]
        except KeyError:
            raise Exception("Unknown dataset ! Check the name again")

    mapping = {
        DATASETS.ModelNet10PCD: (ModelNet10PCD, MN10_CLASS_DICT),
        DATASETS.ModelNet10OFF: (ModelNet10OFF, MN10_CLASS_DICT),
        DATASETS.ModelNet10PLY: (ModelNet10PLY, MN10_CLASS_DICT),

        DATASETS.ModelNet40PCD: (ModelNet40PCD, MN40_CLASS_DICT),
        DATASETS.ModelNet40OFF: (ModelNet40OFF, MN40_CLASS_DICT),
        DATASETS.ModelNet40PLY: (ModelNet40PLY, MN40_CLASS_DICT),

        DATASETS.ScanNet: (ScanNet, SN_CLASS_DICT),
        DATASETS.ScanNetToModelNet10: (ScanNetToModelNet10, MN10_CLASS_DICT),
        DATASETS.ScanNetToModelNet40: (ScanNetToModelNet40, MN40_CLASS_DICT),

        DATASETS.ShapeNetSemToScanNetDet: (ShapeNetSemToScanNetDet, SNS_CLASS_DICT),

        DATASETS.ScanObjectNN: (ScanObjectNN, SONN_CLASS_DICT),
        DATASETS.ScanObjectNN_T25: (ScanObjectNN_T25, SONN_CLASS_DICT),
        DATASETS.ScanObjectNN_T50_RS: (ScanObjectNN_T50_RS, SONN_CLASS_DICT),
        DATASETS.ScanObjectNNToModelNet: (ScanObjectNNToModelNet, SONN_CLASS_DICT_COMBINED),
        DATASETS.ScanObjectNN_T25ToModelNet: (ScanObjectNN_T25ToModelNet, SONN_CLASS_DICT_COMBINED),
        DATASETS.ScanObjectNN_T50_RSToModelNet: (ScanObjectNN_T50_RSToModelNet, SONN_CLASS_DICT_COMBINED),
    }

    try:
        return mapping[dataset_name]
    except KeyError:
        raise Exception("Unknown dataset ! Check the name again")
