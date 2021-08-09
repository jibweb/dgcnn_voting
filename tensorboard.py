#!/usr/bin/env python2.7

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="port to run on")
    parser.add_argument("-f", "--exp_folder", help="Choose experiment folder")
    args = parser.parse_args()

    if args.exp_folder:
        SAVE_DIR = "output_save/" + args.exp_folder + "/"
    else:
        with open(".experiment_history") as fp:
            exp_folders = fp.readlines()
        SAVE_DIR = "output_save/" + exp_folders[-1].strip() + "/"

    logdir_opts = "--logdir=train:{}train_tb/".format(SAVE_DIR)
    logdir_opts += ",val:{}val_tb/".format(SAVE_DIR)
    port_opts = ""
    if args.port:
        port_opts = "--port {}".format(args.port)

    called = ["tensorboard", logdir_opts, port_opts]
    print "STARTED:", " ".join(called)
    os.system(" ".join(called))
