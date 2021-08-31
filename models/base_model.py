from enum import Enum
import numpy as np
import tensorflow as tf
from utils.tf import fc, fc_bn
from utils.params import params as p

from convolution_ops import g_2d_k, voting_module_simple, sample_and_group, group_point


LOCAL_FEATS_BLOCK = Enum("LOCAL_FEATS_BLOCK",
                         ["PointNet",])

FEATS_COMBI_BLOCK = Enum("FEATS_COMBI_BLOCK",
                         ["KPConv", "SKPConv", "GAT", "EdgeConv", "DynEdgeConv"])

POOLING_BLOCK = Enum("POOLING_BLOCK",
                     ["MaxPool", "SingleNode", "VoteMaxPool"])


class Model(object):
    def __init__(self, local_feats, feats_combi, pooling,
                 bn_decay=None):
        self.local_feats_block = local_feats
        self.feats_combi_block = feats_combi
        self.pooling_block = pooling

        # --- I/O Tensors -----------------------------------------------------
        self.support_pts = tf.placeholder(tf.float32,
                                          (None,
                                           3),
                                          name="support_pts")

        self.lrf_transforms = tf.placeholder(tf.float32,
                                             [None,
                                              3, 3],
                                             name="lrf_transforms")

        if p.wnormals:
            self.pt_feats = tf.placeholder(tf.float32,
                                           (None,
                                            p.max_support_point,
                                            p.region_sample_size,
                                            6),
                                           name="pt_feats")
        else:
            self.pt_feats = tf.placeholder(tf.float32,
                                           (None,
                                            p.max_support_point,
                                            p.region_sample_size,
                                            3),
                                           name="pt_feats")

        self.valid_pts = tf.placeholder(tf.float32,
                                        (None,
                                         p.max_support_point,
                                         1),
                                        name="valid_pts")

        self.scales = tf.placeholder(tf.float32,
                                        (None,
                                         p.max_support_point,
                                         1),
                                        name="scales")

        if p.with_height:
            self.heights = tf.placeholder(tf.float32,
                                        (None,
                                         p.max_support_point,
                                         1),
                                        name="heights")

        if p.with_bbox:
            self.bbox_centers_gt = tf.placeholder(tf.float32,
                                                  (None, 3),
                                                  name="bbox_centers_gt")
            self.bbox_extent_gt = tf.placeholder(tf.float32,
                                                 (None, 3),
                                                 name="bbox_extent_gt")

        self.y = tf.placeholder(tf.float32,
                                [None, p.num_classes],
                                name="y")

        self.mask = tf.placeholder(tf.float32,
                                       [None],
                                       name="mask")

        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.pool_drop = tf.placeholder(tf.float32, name="pool_drop_prob")
        self.bn_decay = bn_decay


    def get_base_feed_dict(self, x_batch, y_batch, is_training):
        xb_support_pts = np.array([x_i[0] for x_i in x_batch]).reshape((-1, 3))
        xb_lrf_transforms = np.array([x_i[2] for x_i in x_batch]).reshape((-1, 3, 3))
        xb_valid_pts = np.array([x_i[3] for x_i in x_batch]).reshape((-1, p.max_support_point, 1))
        xb_scales = np.array([x_i[5] for x_i in x_batch]).reshape((-1, p.max_support_point, 1))
        xb_heights = np.array([x_i[6] for x_i in x_batch]).reshape((-1, p.max_support_point, 1))
        xb_bbox = np.array([x_i[7] for x_i in x_batch]).reshape((-1, 6))
        xb_pt_feats = np.array([x_i[4] for x_i in x_batch])
        xb_mask = [item for x_i in x_batch for item in x_i[3]]

        if self.pooling_block == POOLING_BLOCK.SingleNode:
            y_batch = [val for val in y_batch for i in range(p.max_support_point)]
        elif self.pooling_block == POOLING_BLOCK.VoteMaxPool:
            y_batch = np.reshape(y_batch, [-1, 1, p.num_classes])
            y_batch = np.tile(y_batch, [p.num_cluster, 1])
            y_batch = np.reshape(y_batch, [-1, p.num_classes])


        pool_drop = p.pool_drop_prob if is_training else 0.

        feed_dict =  {
            self.support_pts: xb_support_pts,
            self.lrf_transforms: xb_lrf_transforms,
            self.valid_pts: xb_valid_pts,
            self.scales: xb_scales,
            self.pt_feats: xb_pt_feats,
            self.mask: xb_mask,
            self.y: y_batch,
            self.pool_drop: pool_drop,
            self.is_training: is_training,
        }

        if p.with_height:
            feed_dict[self.heights] = xb_heights

        if p.with_bbox:
            feed_dict[self.bbox_centers_gt] = xb_bbox[:, :3]
            feed_dict[self.bbox_extent_gt] = xb_bbox[:, 3:]

        return feed_dict

    def get_local_feats(self):
        local_feats = self.pt_feats
        # --- Local feats -----------------------------------------------------
        print "local_feats/IN:",  local_feats.get_shape()

        with tf.variable_scope('local_feats'):
            if self.local_feats_block == LOCAL_FEATS_BLOCK.PointNet:
                for i in range(len(p.local_feats_layers)):
                    local_feats = g_2d_k(local_feats,
                                         "pn_" + str(i),
                                         p.local_feats_layers[i],
                                         self.is_training, self.bn_decay,
                                         p.reg_constant)

            print "local_feats/b:",  local_feats.get_shape()

            local_feats = tf.reduce_max(local_feats, axis=2,
                                            name='max_g')

        print "local_feats/c:", local_feats.get_shape()
        local_feats = tf.reshape(local_feats, [-1,
                                               p.local_feats_layers[-1]])

        if p.with_height:
            heights = tf.reshape(self.heights, [-1, 1])
            local_feats = tf.concat([local_feats, heights], -1)

        print "local_feats/OUT:", local_feats.get_shape()
        return local_feats

    def get_pooling(self, pooling_feats):
        # --- Pooling ---------------------------------------------------------
        print "pooling/IN:",  pooling_feats.get_shape()

        with tf.variable_scope('pooling'):
            if self.pooling_block == POOLING_BLOCK.SingleNode:
                pooling_feats = tf.reshape(pooling_feats,
                                           [-1, p.max_support_point,
                                            p.feats_combi_layers[-1]])
                # Zeroing-out the features of the invalid points
                pooling_feats = tf.multiply(pooling_feats, self.valid_pts)
                pooling_feats = tf.reshape(pooling_feats,
                                           [-1, p.feats_combi_layers[-1]])

            if self.pooling_block == POOLING_BLOCK.MaxPool:
                with tf.variable_scope('maxpool'):
                    pooling_feats = tf.reshape(pooling_feats,
                                               [-1, p.max_support_point,
                                                p.feats_combi_layers[-1]])

                    # Zeroing-out the features of the invalid points
                    pooling_feats = tf.multiply(pooling_feats, self.valid_pts)
                    pooling_feats = tf.reduce_max(pooling_feats, axis=1, name='max_pool')

            elif self.pooling_block == POOLING_BLOCK.VoteMaxPool:
                with tf.variable_scope('vote'):
                    pooling_feats = tf.reshape(pooling_feats,
                                               [-1, p.feats_combi_layers[-1]])
                    scales = tf.reshape(self.scales, [-1, 1])
                    vote_xyz, vote_feats = voting_module_simple(
                        self.support_pts,
                        pooling_feats,
                        self.lrf_transforms,
                        scales,
                        self.is_training,
                        use_batch_norm=p.use_batch_norm,
                        batch_norm_momentum=p.batch_norm_momentum,
                        reg_constant=p.reg_constant)

                    self.vote_xyz = tf.reshape(vote_xyz,
                                               [-1, p.max_support_point, 3])
                    # self.vote_xyz = tf.concat(
                    #     [self.vote_xyz[:,:,:-1],
                    #      tf.constant(np.zeros((p.batch_size, p.max_support_point, 1),
                    #                  dtype=np.float32))],
                    #     -1)


                    # Put all the votes together far away
                    # This ensures farthest_point_sampling will group them
                    self.vote_xyz = tf.where(
                        tf.tile(tf.cast(self.valid_pts, tf.bool), [1, 1, 3]),
                        self.vote_xyz,
                        1e6*np.ones((p.batch_size, p.max_support_point, 3)))

                with tf.variable_scope('maxpool'):
                    feats_obj = tf.reshape(vote_feats,
                                           [-1, p.max_support_point,
                                            p.feats_combi_layers[-1]])

                    # Zeroing-out the features of the invalid points
                    feats_obj = tf.multiply(feats_obj, self.valid_pts)

                    grouped_feats, centroid_idx, self.cluster_count, idx = sample_and_group(
                        self.vote_xyz,
                        feats_obj,
                        p.num_cluster,
                        p.max_support_point,
                        p.group_radius)

                    print "pooling_block/clusters_feats:",  grouped_feats.get_shape()
                    print "pooling_block/cluster_count:", self.cluster_count.get_shape()

                    # Check validity
                    self.cluster_validity = tf.reduce_mean(
                        group_point(self.valid_pts, idx), axis=2)
                    self.cluster_validity = tf.cast(self.cluster_validity,
                                                    tf.bool)
                    self.cluster_validity = tf.reshape(self.cluster_validity,
                                                       [-1, p.num_cluster])

                    # Max-pooling within each cluster
                    pooling_feats = tf.reduce_max(grouped_feats, axis=2,
                                                  name='max_in_cluster')
                    pooling_feats = tf.reshape(pooling_feats,
                                               [-1, p.feats_combi_layers[-1]])

                if p.with_bbox:
                    with tf.variable_scope('bbox'):
                        # Need to first only account for unique elements in the center
                        #  before reduce to the mean
                        parts_center = tf.reshape(self.support_pts,
                                                  [-1, p.max_support_point, 3])
                        self.cluster_points = group_point(parts_center, idx)
                        cluster_means = tf.reduce_mean(self.cluster_points, axis=2)
                        centered_cluster_points = self.cluster_points - tf.expand_dims(cluster_means, 2)
                        pn_pool_feats = tf.concat([grouped_feats, centered_cluster_points], -1)
                        with tf.variable_scope('pn_pool'):
                            pn_pool_layers = [8, 8, 16, 128]
                            for i in range(len(pn_pool_layers)):
                                pn_pool_feats = g_2d_k(pn_pool_feats,
                                                       "pn_" + str(i),
                                                       pn_pool_layers[i],
                                                       self.is_training, self.bn_decay,
                                                       p.reg_constant)

                            pn_pool_feats = tf.reduce_max(pn_pool_feats, axis=2,
                                                        name='max_g')
                            pn_pool_feats = tf.reshape(pn_pool_feats,
                                               [-1, pn_pool_layers[-1]])
                            bbox_preds = fc_bn(pn_pool_feats, pn_pool_layers[-1],
                                               scope='bbox_pred_1',
                                               is_training=self.is_training,
                                               bn_decay=self.bn_decay,
                                               reg_constant=p.reg_constant)
                            bbox_preds = fc_bn(bbox_preds, 6,
                                               scope='bbox_pred_2',
                                               is_training=self.is_training,
                                               bn_decay=self.bn_decay,
                                               reg_constant=p.reg_constant)
                            bbox_preds = tf.reshape(bbox_preds, (-1, p.num_cluster, 6))

                            self.bbox_centers = tf.reshape(bbox_preds[..., :3] + cluster_means, (-1, p.num_cluster, 3))
                            self.bbox_extent = tf.reshape(bbox_preds[..., 3:], (-1, p.num_cluster, 3))

                        print "pooling_block/cluster_points:",  self.cluster_points.get_shape()

                        # Re-orient the points of the part
                        # transposed_lrf = tf.transpose(self.lrf_transforms, perm=[0, 2, 1])
                        # parts_points = tf.reshape(self.pt_feats[:, :, :, :3],
                        #                           [-1, p.region_sample_size, 3])
                        # parts_points = tf.matmul(parts_points, transposed_lrf)

                        # # Re-scale and translate back to the global RF
                        # parts_points = tf.reshape(parts_points,
                        #                           [-1, p.max_support_point,
                        #                            p.region_sample_size, 3])
                        # parts_points /= tf.expand_dims(self.scales, 2) + 1e-6
                        # parts_points += tf.expand_dims(parts_center, 2)

                        # # Bbox for each part
                        # xyz_min = tf.reduce_min(parts_points, axis=2)
                        # xyz_max = tf.reduce_max(parts_points, axis=2)

                        # grouped_min = group_point(xyz_min, idx)
                        # grouped_max = group_point(xyz_max, idx)
                        # self.cluster_extent = tf.reduce_max(grouped_max, axis=2) - \
                        #     tf.reduce_min(grouped_min, axis=2)
                        # print "pooling_block/cluster_extent:", self.cluster_extent.get_shape()

        print "pooling_block/OUT:",  pooling_feats.get_shape()

        # --- Classification --------------------------------------------------
        with tf.variable_scope('classification'):
            fcls = fc_bn(pooling_feats, p.clf_layers[0],
                         scope='fc_1',
                         is_training=self.is_training,
                         bn_decay=self.bn_decay,
                         reg_constant=p.reg_constant)
            fcls = tf.nn.dropout(fcls, 1.0 - self.pool_drop)

            fcls = fc_bn(fcls, p.clf_layers[1],
                         scope='fc_2',
                         is_training=self.is_training,
                         bn_decay=self.bn_decay,
                         reg_constant=p.reg_constant)
            fcls = tf.nn.dropout(fcls, 1.0 - self.pool_drop)

            return fc(fcls, p.num_classes,
                      activation_fn=None, scope='logits')

    def get_cluster_center(self, cluster_points, cluster_count):
        """
            Numpy function. Compute the mean of each cluster only
            accounting for different points in the cluster, ignoring
            the repetitions of the first element,
        """
        return np.array(
            [[np.mean(b[c_idx, :cluster_count[b_idx, c_idx]], axis=0)
              for c_idx in range(p.num_cluster)]
             for b_idx, b in enumerate(cluster_points)])

    def get_bbox(self, cluster_inference, cluster_points, cluster_extent,
                 cluster_count, cluster_validity):
        cluster_centers = self.get_cluster_center(cluster_points, cluster_count)
        probs = cluster_inference.reshape([-1, p.num_cluster, p.num_classes])
        pred_cls = np.argmax(probs, axis=2)
        probs = np.exp(probs - np.max(probs, axis=2, keepdims=True))
        probs /= np.sum(probs, axis=2, keepdims=True)

        return [
            [[probs[b_idx, c_idx][pred_cls[b_idx, c_idx]]]
                + list(cluster_centers[b_idx, c_idx])
                + list(cluster_extent[b_idx, c_idx])
                + [pred_cls[b_idx, c_idx]]
                for c_idx, c_valid in enumerate(cluster_mask) if c_valid]
            for b_idx, cluster_mask in enumerate(cluster_validity)
        ]

    def get_bbox2(self, cluster_inference, cluster_centers, cluster_extent, cluster_validity):
        probs = cluster_inference.reshape([-1, p.num_cluster, p.num_classes])
        pred_cls = np.argmax(probs, axis=2)
        probs = np.exp(probs - np.max(probs, axis=2, keepdims=True))
        probs /= np.sum(probs, axis=2, keepdims=True)

        return [
            [[probs[b_idx, c_idx][pred_cls[b_idx, c_idx]]]
                + list(cluster_centers[b_idx, c_idx])
                + list(cluster_extent[b_idx, c_idx])
                + [pred_cls[b_idx, c_idx]]
                for c_idx, c_valid in enumerate(cluster_mask) if c_valid]
            for b_idx, cluster_mask in enumerate(cluster_validity)
        ]

    def get_base_loss(self):
        #  --- Cross-entropy loss ---------------------------------------------
        with tf.variable_scope('cross_entropy'):
            if self.pooling_block == POOLING_BLOCK.VoteMaxPool:
                y = tf.reshape(self.y, [-1, p.num_cluster, p.num_classes])
                self.y_mask = tf.boolean_mask(y, self.cluster_validity,
                                    name='y_mask')
                self.inference_mask = tf.boolean_mask(tf.reshape(self.inference,
                                                                 [-1,
                                                                  p.num_cluster,
                                                                  p.num_classes]),
                                                      self.cluster_validity,
                                                      name="inference_mask")
                diff = tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.y_mask,
                        logits=self.inference_mask)

            else:
                diff = tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.y,
                        logits=self.inference)
                if self.pooling_block == POOLING_BLOCK.SingleNode:
                    diff = tf.boolean_mask(diff, self.mask, name='boolean_mask')

            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy_avg', cross_entropy)

        # --- L2 Regularization -----------------------------------------------
        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization_loss_avg', reg_loss)

        total_loss = cross_entropy + reg_loss

        # --- Vote loss -------------------------------------------------------
        if self.pooling_block == POOLING_BLOCK.VoteMaxPool:
            # Vote loss
            votes_dist = tf.reshape(tf.reduce_sum(tf.square(self.vote_xyz), axis=-1), [-1])
            votes_dist_masked = tf.boolean_mask(votes_dist, self.mask, name='vote_boolean_mask')
            vote_loss = tf.reduce_mean(votes_dist_masked)
            tf.summary.scalar('vote_loss_avg', vote_loss)
            total_loss += p.vote_loss_weight * vote_loss

            if p.with_bbox:
                bbox_centers_gt = tf.tile(tf.reshape(self.bbox_centers_gt, (-1, 1, 3)), [1, p.num_cluster, 1])
                bbox_extent_gt = tf.tile(tf.reshape(self.bbox_extent_gt, (-1, 1, 3)), [1, p.num_cluster, 1])

                bbox_centers_loss = tf.reduce_sum(tf.abs(self.bbox_centers - bbox_centers_gt), axis=-1)
                bbox_extent_loss = tf.reduce_sum(tf.abs(self.bbox_extent - bbox_extent_gt), axis=-1)

                bbox_centers_loss = tf.boolean_mask(bbox_centers_loss,
                                                    self.cluster_validity,
                                                    name="bbox_centers_loss_mask")
                bbox_extent_loss = tf.boolean_mask(bbox_extent_loss,
                                                   self.cluster_validity,
                                                   name="bbox_extent_loss_mask")
                bbox_centers_loss = tf.reduce_mean(bbox_centers_loss)
                bbox_extent_loss = tf.reduce_mean(bbox_extent_loss)

                tf.summary.scalar('bbox_centers_loss_avg', bbox_centers_loss)
                tf.summary.scalar('bbox_extent_loss_avg', bbox_extent_loss)

                total_loss += bbox_centers_loss
                total_loss += bbox_extent_loss

        tf.summary.scalar('total_loss', total_loss)
        return total_loss
