import os
import random
from glob import glob

import h5py
import numpy as np
import trimesh
# from joblib import delayed, Parallel

from dataset import get_dataset, DATASETS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_hdf5(fns, labels, dataset_name, rescaling=False, random_orientation=False, scale_from_file=False, sonn_with_bg=False):
    # Load files and sample points
    dataset_points = []
    for fn in fns:
        try:
            mesh = trimesh.load_mesh(fn)
            points = mesh.vertices
            points = np.hstack([points[:,:1], points[:,2:3], points[:, 1:2]])
        except:
            # print fn
            if fn[-3:] == "bin":
                bindata = np.fromfile(fn, dtype=np.float32)
                points = bindata[1:].reshape((-1, 11))
                if sonn_with_bg:
                    points = np.array(points[:,0:3])
                else:
                    ##To remove backgorund points
                    ##filter unwanted class
                    filtered_idx = np.intersect1d(np.intersect1d(np.where(points[:,-1]!=0)[0],np.where(points[:,-1]!=1)[0]), np.where(points[:,-1]!=2)[0])
                    (values, counts) = np.unique(points[filtered_idx,-1], return_counts=True)
                    max_ind = np.argmax(counts)
                    idx = np.where(points[:,-1]==values[max_ind])[0]
                    points = np.array(points[idx,0:3])

                # points = np.hstack([points[:,:1], points[:,2:3], points[:, 1:2]])

        if rescaling:
            points -= np.mean(points, axis=0)
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale


        dataset_points.append(points[(points.shape[0] * np.random.random(2048)).astype(int)])

    if not os.path.exists(os.path.dirname(dataset_name)):
        os.makedirs(os.path.dirname(dataset_name))

    # Create the HDF5 file
    print "Creating dataset", dataset_name
    dataset = h5py.File(dataset_name, "w")

    # Fill in the HDF5 file
    data = dataset.create_dataset("data", shape=[len(dataset_points), 2048, 3])
    dataset['data'][:] = np.array(dataset_points)

    label = dataset.create_dataset("label", shape=[len(dataset_points), 1])
    dataset['label'][:] = np.reshape(labels, newshape=[len(dataset_points), 1])

    dataset.close()


def create_h5(dataset_to_load, dataset_basename, skip_train=False, rescaling=False, random_orientation=False, scale_from_file=False, sonn_with_bg=False):
    file_length = 2048
    # Create the HDF5 files for the train set

    Dataset, CLASS_DICT = get_dataset(dataset_to_load)
    d = Dataset()
    d = Dataset(balance_train_set=False,
                balance_test_set=False,
                balance_val_set=False,
                shuffle_class=False,
                one_hot=False,
                val_set_pct=0.)
    d.prepare_sets()

    if d.train_x and not skip_train:
        dataset_names = []
        for file_idx in range(int(len(d.train_x) / file_length)):
            print('train file %d / %d' % (file_idx, int(len(d.train_x) / file_length)))

            dataset_name = 'data/{}/train{}.h5'.format(dataset_basename, file_idx)
            dataset_names.append(dataset_name)
            dataset_name = os.path.join(BASE_DIR, dataset_name)

            create_hdf5(d.train_x[file_idx * file_length:(file_idx + 1) * file_length],
                        d.train_y[file_idx * file_length:(file_idx + 1) * file_length],
                        dataset_name, rescaling=rescaling, random_orientation=random_orientation,
                        scale_from_file=scale_from_file, sonn_with_bg=sonn_with_bg)
        if len(d.train_x) % file_length != 0:
            remaining = len(d.train_x) % file_length
            dataset_name = 'data/{}/train{}.h5'.format(dataset_basename, len(dataset_names))
            dataset_names.append(dataset_name)
            dataset_name = os.path.join(BASE_DIR, dataset_name)
            create_hdf5(d.train_x[len(d.train_x) - remaining:len(d.train_x)],
                        d.train_y[len(d.train_x) - remaining:len(d.train_x)],
                        dataset_name, rescaling=rescaling, random_orientation=random_orientation,
                        scale_from_file=scale_from_file, sonn_with_bg=sonn_with_bg)
        with open("data/{}/train_files.txt".format(dataset_basename), "w") as fp:
            for dname in dataset_names:
                fp.write(dname + "\n")

    if d.test_x:
        # Create the HDF5 files for the test set
        dataset_names = []
        for file_idx in range(int(len(d.test_x) / file_length)):
            print('test file %d / %d' % (file_idx, int(len(d.test_x) / file_length)))

            dataset_name = 'data/{}/test{}.h5'.format(dataset_basename, file_idx)
            dataset_names.append(dataset_name)
            dataset_name = os.path.join(BASE_DIR, dataset_name)

            create_hdf5(d.test_x[file_idx * file_length:(file_idx + 1) * file_length],
                        d.test_y[file_idx * file_length:(file_idx + 1) * file_length],
                        dataset_name, rescaling=rescaling, random_orientation=random_orientation,
                        scale_from_file=scale_from_file, sonn_with_bg=sonn_with_bg)
        if len(d.test_x) % file_length != 0:
            remaining = len(d.test_x) % file_length
            dataset_name = 'data/{}/test{}.h5'.format(dataset_basename, len(dataset_names))
            dataset_names.append(dataset_name)
            dataset_name = os.path.join(BASE_DIR, dataset_name)
            create_hdf5(d.test_x[len(d.test_x) - remaining:len(d.test_x)],
                        d.test_y[len(d.test_x) - remaining:len(d.test_x)],
                        dataset_name, rescaling=rescaling, random_orientation=random_orientation,
                        scale_from_file=scale_from_file, sonn_with_bg=sonn_with_bg)
        with open("data/{}/test_files.txt".format(dataset_basename), "w") as fp:
            for dname in dataset_names:
                fp.write(dname + "\n")


if __name__ == "__main__":
    dataset_to_h5 = "ScanObjectNNToModelNet"
    sonn_with_bg = True
    rescaling = False
    skip_train = True

    dataset_savename = [dataset_to_h5]
    if sonn_with_bg:
        dataset_savename.append("w_bg")
    if rescaling:
        dataset_savename.append("rescaled")


    create_h5(dataset_to_h5, "_".join(dataset_savename), rescaling=rescaling, sonn_with_bg=sonn_with_bg, skip_train=skip_train)
