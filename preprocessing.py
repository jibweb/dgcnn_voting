from enum import Enum
from functools import partial
import numpy as np
import sys

from utils.params import params as p

sys.path.insert(0, "./build/src")
from py_graph_construction import PyGraph


LRFS = Enum("LRFS", "EYELRF PCALRF ZLRF")
# "EYELRF"  : 1
# "PCALRF"  : 2
# "ZLRF"    : 3

for l in LRFS:
    print l, l.value

# Graph structure
p.define("max_support_point", 1024)
p.define("neigh_size", 100.)
p.define("neigh_nb", 30)
p.define("fill_neighbors_w_self", False)
p.define("region_sample_size", 64)
p.define("wnormals", False)
p.define("wregions", True)
p.define("lrf", "PCALRF")
# p.define("shadowing_threshold", 10.)
p.define("disconnect_rate", 0.)
p.define("bias_mat", False)
# p.define("keep_scaled_parts", False)
p.define("height_to_zero", False)

p.define("sonn_no_bg", False)

# Data Augmentation
p.define("data_augmentation", True)
p.define("normal_noise", 0.)
p.define("normal_smoothing", False)
p.define("normal_occlusion", -2.)
p.define("rescaling", False)
p.define("z_rotation", False)

p.define("debug", False)


def neighbors_to_bias(neighbors):
    """
     Prepare adjacency matrix by converting it to bias vectors.
     Expected shape: [nodes, nodes]
     Originally from github.com/PetarV-/GAT
    """
    adj = np.eye(neighbors.shape[0])
    for idx, neigh in enumerate(neighbors):
        adj[neigh[neigh != -1], idx] = 1
        adj[idx, neigh[neigh != -1]] = 1
    return -1e9 * (1.0 - adj)



def graph_prepare(fn, debug, lrf_code, wnormals, no_bg, height_to_zero):
    graph = PyGraph(debug=debug, lrf=lrf_code, wnormals=wnormals)

    file_ext = fn[-3:].strip().lower()
    if file_ext == "ply":
        graph.initialize_from_ply(fn, height_to_zero)
    elif file_ext == "pcd":
        graph.initialize_from_pcd(fn, fn[:-3] + "adj")
    elif file_ext == "bin":
        if no_bg:
            graph.initialize_from_scanobjectnn(fn, fn[:-3] + "adj", 1, height_to_zero=height_to_zero)
        else:
            graph.initialize_from_scanobjectnn(fn, fn[:-3] + "adj", height_to_zero=height_to_zero)

    return graph


def graph_process(fn, p, with_fn, lrf):
    graph = graph_prepare(fn, p.debug, lrf[p.lrf].value, p.wnormals, p.sonn_no_bg, p.height_to_zero)
    if p.data_augmentation:
        graph.data_augmentation(rescaling=p.rescaling,
                                z_rotation=p.z_rotation,
                                normal_smoothing=p.normal_smoothing,
                                normal_occlusion=p.normal_occlusion,
                                normal_noise=p.normal_noise)

    if p.wregions:
        res = graph.sample_support_points_and_regions(p.neigh_size,
                                                      max_support_point=p.max_support_point,
                                                      neighbors_nb=p.neigh_nb,
                                                      fill_neighbors_w_self=p.fill_neighbors_w_self,
                                                      region_sample_size=p.region_sample_size,
                                                      disconnect_rate=int(100*p.disconnect_rate))
    else:
        raise Exception("wregions is no longer a valid option")
        # res = graph.sample_support_points(p.neigh_size,
        #                                   max_support_point=p.max_support_point,
        #                                   neighbors_nb=p.neigh_nb,
        #                                   shadowing_threshold=p.shadowing_threshold)

    # if p.keep_scaled_parts:
    #     support_points, neigh_indices, lrf_transforms, valid_indices, feats, scales = res
    #     scales[scales == 0.] = 1.
    #     feats /= scales.reshape((p.max_support_point, 1, 1))
    #     scales = np.ones((p.max_support_point))
    #     res = (support_points, neigh_indices, lrf_transforms, valid_indices, feats, scales)

    if with_fn:
        res += (fn,)

    if p.bias_mat:
        return (res[0], neighbors_to_bias(res[1])) + res[2:]

    return res


def get_processing_func(p, with_fn=False):
    return partial(graph_process, p=p, with_fn=with_fn, lrf=LRFS)
