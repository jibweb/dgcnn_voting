#!/usr/bin/env python2.7

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import sys
import tensorflow as tf

from dataset import get_dataset, DATASETS
from models import get_model
from preprocessing import get_processing_func
from utils.logger import log, set_log_level
from utils.params import params as p
from utils.viz import plot_confusion_matrix

# ---- Parameters ----
p.define("test_dataset", DATASETS.ModelNet40PLY.name)
p.define("test_ckpt", 10)
p.define("test_repeat", 3)

# Generic
set_log_level("INFO")


def compute_metrics(cm):
    cls_count = np.sum(cm, axis=1)
    indices = cls_count != 0.

    # Class-level metrics
    cls_recall = 100. * cm.diagonal() / cls_count

    # Average metrics
    avg_recall = np.mean(cls_recall[indices])
    accuracy = 100. * np.sum(cm.diagonal()) / np.sum(cm)

    return accuracy, avg_recall, cls_recall


if __name__ == "__main__":
    # === SETUP ===============================================================
    # --- Parse arguments for specific params file ----------------------------
    parser = argparse.ArgumentParser("Test a given model on a given dataset")
    parser.add_argument("-f", "--exp_folder", help="Choose experiment folder")
    parser.add_argument("-d", "--dataset", help="Specify a different dataset")
    parser.add_argument("-v", "--viz", action="store_true",
                        help="Show Confusion Matrix")
    p.add_arguments(parser)
    args = parser.parse_args()

    if args.exp_folder:
        # SAVE_DIR = "output_save/" + args.exp_folder + "/"
        SAVE_DIR = args.exp_folder + "/"
    else:
        if os.path.isfile(".experiment_history"):
            with open(".experiment_history") as fp:
                exp_folders = fp.readlines()
            SAVE_DIR = "output_save/" + exp_folders[-1].strip() + "/"
        else:
            log("No experiment was previously run. " +
                "Please specify from which folder to restore using -f\n")
            sys.exit()

    p.define_from_file("{}/params.yaml".format(SAVE_DIR))
    p.load_from_parser(args)

    # --- Model Checkpoint definition -----------------------------------------
    if not args.test_ckpt:
        epochs_nb_found = sorted([int(dirname[6:])
                                  for dirname in os.listdir(SAVE_DIR)
                                  if dirname[:5] == "model"])
        if len(epochs_nb_found) != 0:
            p.test_ckpt = epochs_nb_found[-1]
        else:
            log("Failed to find a trained model to restore\n")
            sys.exit()

    MODEL_CKPT = "model_{0}/model.ckpt-{0}".format(p.test_ckpt)

    # --- Pre processing function setup ---------------------------------------
    p.data_augmentation = False

    # FORCE rescale
    # p.data_augmentation = True
    # p.z_rotation = False
    # p.rescaling = True

    feat_compute = get_processing_func(p, with_fn=True)

    # --- Dataset setup -------------------------------------------------------
    if not args.test_dataset:
        p.test_dataset = p.dataset

    Dataset, CLASS_DICT = get_dataset(p.test_dataset)
    dataset = Dataset(batch_size=p.batch_size,
                      val_set_pct=p.val_set_pct)

    # --- Model Setup ---------------------------------------------------------
    Model = get_model(
        local_feats=p.local_feats,
        feats_combi=p.feats_combi,
        pooling=p.pooling)
    model = Model()

    saver = tf.train.Saver()

    # === GRAPH COMPUTATION ===================================================
    exp_hash = p.get_hash()
    with tf.Session() as sess:
        saver.restore(sess, SAVE_DIR + MODEL_CKPT)

        # Testing
        log("Setup finished, starting testing now ... \n\n")
        print "Test experiment hash:", exp_hash
        print "Parameters:"
        print p, "\n"
        total_cm = np.zeros((p.num_classes, p.num_classes), dtype=np.int32)
        total_obj_cm = np.zeros((p.num_classes, p.num_classes), dtype=np.int32)
        total_clust_cm = np.zeros((p.num_classes, p.num_classes), dtype=np.int32)
        misclassified_fns = set()

        for repeat in range(p.test_repeat):
            for xs, ys in dataset.test_batch(process_fn=feat_compute):
                if "votemaxpool" == p.pooling.lower():
                    obj_preds, y_pred, y_true, obj_mask, loss = sess.run(
                        [model.inference,
                         model.inference_mask,
                         model.y_mask,
                         model.cluster_validity,
                         model.loss],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))
                else:
                    preds, loss = sess.run(
                        [model.inference,
                         model.loss],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))

                    if "singlenode" == p.pooling.lower():
                        ys = np.array([val for val in ys for i in range(p.max_support_point)])
                        mask = np.array([item for x_i in xs for item in x_i[3]],
                                        dtype=bool)
                        y_true = ys[mask]
                        y_pred = preds[mask]
                    else:
                        y_true = np.array(ys)
                        y_pred = preds

                y_true = np.argmax(y_true, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
                cm = confusion_matrix(y_true, y_pred,
                                      labels=range(p.num_classes))

                # --- Object/Cluster level metric -----------------------------
                if "votemaxpool" == p.pooling.lower():
                    y_obj_true = np.argmax(np.array(ys), axis=1)
                    obj_preds = obj_preds.reshape((-1, p.num_cluster,
                                                   p.num_classes))

                    # Biggest cluster
                    y_clust_pred = [np.argmax(np.max(pred[obj_mask[i]], axis=0))
                                    for i, pred in enumerate(obj_preds)]

                    clust_cm = confusion_matrix(y_obj_true, y_clust_pred,
                                                labels=range(p.num_classes))
                    total_clust_cm += clust_cm

                elif "singlenode" == p.pooling.lower():
                    obj_mask = mask.reshape((-1, p.max_support_point))
                    obj_preds = preds.reshape((-1, p.max_support_point, p.num_classes))
                    obj_ys = ys.reshape((-1, p.max_support_point, p.num_classes))
                    y_obj_true = np.argmax(obj_ys[:, 0, :], axis=1)

                if p.pooling.lower() in ["singlenode", "votemaxpool"]:
                    y_obj_pred = [np.argmax(np.sum(obj_preds[i][obj_mask[i]], axis=0))
                                  for i in range(len(obj_preds))]

                    obj_cm = confusion_matrix(y_obj_true, y_obj_pred,
                                              labels=range(p.num_classes))
                    total_obj_cm += obj_cm
                    correct_preds = y_obj_true == y_obj_pred
                    #correct_preds = correct_preds.reshape((-1, p.num_cluster))
                    # TODO: np.any means if one cluster gets it, it's fine
                    # np.all means all clusters have to be correct
                    #correct_preds = np.any(correct_preds, axis=1)
                else:
                    correct_preds = y_true == y_pred

                x_fns = np.array([x_i[-1] for x_i in xs])
                misclassified_fns_batch = x_fns[np.logical_not(correct_preds)]
                for fn in misclassified_fns_batch:
                    misclassified_fns.add(fn)

                log("Repeat {}/{} -- acc: {:.1f} | loss: {:.3f}", repeat+1,
                    p.test_repeat, 100*np.mean(correct_preds), loss)
                total_cm += cm

    print ""
    # === METRICS DISPLAY =====================================================
    accuracy, avg_recall, cls_recall = compute_metrics(total_cm)
    print "# Accuracy: {:.2f}".format(accuracy)
    print "# Average Recall: {:.2f}".format(avg_recall)
    for idx, cls_name in enumerate(sorted(CLASS_DICT.keys())):
        print "{:20s}:\t{:.2f}".format(cls_name, cls_recall[idx])

    # --- Object-level metrics ------------------------------------------------
    if p.pooling.lower() in ["singlenode", "votemaxpool"]:
        accuracy, avg_recall, cls_recall = compute_metrics(total_obj_cm)
        print "# Obj Accuracy: {:.2f}".format(accuracy)
        print "# Obj Average Recall: {:.2f}".format(avg_recall)
        for idx, cls_name in enumerate(sorted(CLASS_DICT.keys())):
            print "{:20s}:\t{:.2f}".format(cls_name, cls_recall[idx])

    if p.pooling.lower() == "votemaxpool":
        accuracy, avg_recall, cls_recall = compute_metrics(total_clust_cm)
        print "# Strongest cluster Accuracy: {:.2f}".format(accuracy)
        print "# Strongest cluster Average Recall: {:.2f}".format(avg_recall)
        for idx, cls_name in enumerate(sorted(CLASS_DICT.keys())):
            print "{:20s}:\t{:.2f}".format(cls_name, cls_recall[idx])

    # === RESULTS SAVING ======================================================
    p.save("{}params_{}.yaml".format(SAVE_DIR, exp_hash))
    np.save("{}conf_matrix_{}".format(SAVE_DIR, exp_hash), total_cm)
    with open("{}misclassified_{}.txt".format(SAVE_DIR, exp_hash), "w") as fp:
        fp.write("\n".join(misclassified_fns))

    if p.pooling.lower() in ["singlenode", "votemaxpool"]:
        np.save("{}obj_conf_matrix_{}".format(SAVE_DIR, exp_hash), total_obj_cm)

    if args.viz:
        plot_confusion_matrix(total_cm, sorted(CLASS_DICT.keys()),
                              normalize=True)
        plt.show()

