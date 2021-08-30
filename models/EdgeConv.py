import numpy as np
import tensorflow as tf
from utils.tf import define_scope, conv2d_bn
from utils.params import params as p

from base_model import Model


def get_rel_features(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

    Returns:
    rel position: (batch_size, num_points, k, num_dims)
    """
    point_cloud_central = point_cloud
    point_cloud_shape = point_cloud.get_shape()
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[3].value

    idx_ = tf.range(p.batch_size) * num_points
    idx_ = tf.reshape(idx_, [p.batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    return point_cloud_neighbors, point_cloud_central


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    """
    # og_batch_size = point_cloud.get_shape().as_list()[0]
    # point_cloud = tf.squeeze(point_cloud)
    # if og_batch_size == 1:
    #     point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    # print "get_edge_feature/A-point_cloud:",  point_cloud.get_shape()
    # print "get_edge_feature/A-nn_idx:",  nn_idx.get_shape()
    # print "get_edge_feature/A-point_cloud_central:",  point_cloud_central.get_shape()

    point_cloud_shape = point_cloud.get_shape()
    # batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[3].value

    idx_ = tf.range(p.batch_size) * num_points
    idx_ = tf.reshape(idx_, [p.batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    # print "get_edge_feature/B-point_cloud_neighbors:",  point_cloud_neighbors.get_shape()
    # point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat([point_cloud_central,
                              point_cloud_neighbors-point_cloud_central],
                             axis=-1)
    return edge_feature


class EdgeConv(Model):
    def __init__(self, local_feats, feats_combi, pooling,
                 bn_decay=None):
        super(EdgeConv, self).__init__(local_feats=local_feats,
                                       feats_combi=feats_combi,
                                       pooling=pooling,
                                       bn_decay=bn_decay)

        self.neighbor_indices = tf.placeholder(tf.int32,
                                               [None,
                                                p.max_support_point,
                                                p.neigh_nb],
                                               name="neighbor_indices")

        if not p.fill_neighbors_w_self:
            raise Exception("EdgeConv model requires fill_neighbors_w_self")

        # --- Model properties ------------------------------------------------
        self.inference
        self.loss

    def get_feed_dict(self, x_batch, y_batch, is_training):
        feed_dict = self.get_base_feed_dict(x_batch, y_batch, is_training)
        xb_neigh_indices = np.array([x_i[1] for x_i in x_batch])

        # TODO: cut some edges during training
        # if is_training:
        #     for mat_idx, bias_mat in enumerate(xb_bias_mat):
        #         i_indices, j_indices = np.nonzero(bias_mat == 0.)
        #         for idx in range(len(i_indices)):
        #             if np.random.rand() > p.neigh_disconnect:
        #                 continue
        #             if i_indices[idx] != j_indices[idx]:
        #                 xb_bias_mat[mat_idx][i_indices[idx], j_indices[idx]] = -1e9

        feed_dict[self.neighbor_indices] = xb_neigh_indices

        return feed_dict

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """

        # --- Local feats -----------------------------------------------------
        local_feats = self.get_local_feats()
        local_feats = tf.reshape(local_feats, [-1,
                                               p.max_support_point,
                                               1,
                                               p.local_feats_layers[-1]])

        # --- Feats Combination -----------------------------------------------
        feats_combi = local_feats
        print "feats_combi/IN:",  feats_combi.get_shape()

        neigh_pos, central_pos = get_rel_features(
            tf.reshape(self.support_pts, [-1, p.max_support_point, 1, 3]),
            nn_idx=self.neighbor_indices, k=p.neigh_nb)
        rel_position = neigh_pos - central_pos

        if p.with_rel_scales:
            neigh_scales, central_scales = get_rel_features(
                tf.reshape(self.scales, [-1, p.max_support_point, 1, 1]),
                nn_idx=self.neighbor_indices, k=p.neigh_nb)
            central_scales += 1e-5
            rel_scales = neigh_scales / central_scales

        if p.with_rescaled_pos:
            scales = tf.reshape(self.scales, [-1, p.max_support_point, 1, 1])
            scales = tf.tile(scales, [1, 1, p.neigh_nb, 1])
            scales += 1e-5
            rel_position /= scales

        with tf.variable_scope('feats_combi'):
            for i in range(len(p.feats_combi_layers)):
                with tf.variable_scope('edgeconv_' + str(i)):
                    edge_feature = get_edge_feature(
                        feats_combi,
                        nn_idx=self.neighbor_indices,
                        k=p.neigh_nb)
                    if p.with_rel_scales:
                        edge_feature = tf.concat([edge_feature,
                                                  rel_position,
                                                  rel_scales],
                                                 axis=-1)
                    else:
                        edge_feature = tf.concat([edge_feature,
                                                  rel_position],
                                                 axis=-1)
                    feats_combi = conv2d_bn(edge_feature,
                                            p.feats_combi_layers[i],
                                            [1, 1],
                                            p.reg_constant,
                                            self.is_training,
                                            'edgeconv_' + str(i),
                                            bn_decay=self.bn_decay)
                    feats_combi = tf.reduce_max(feats_combi, axis=-2,
                                                keep_dims=True)

        print "feats_combi/OUT:",  feats_combi.get_shape()

        # --- Pooling ---------------------------------------------------------
        return self.get_pooling(feats_combi)

    @define_scope
    def loss(self):
        return self.get_base_loss()
