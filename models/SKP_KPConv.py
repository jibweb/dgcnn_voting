import numpy as np
import tensorflow as tf
from utils.tf import define_scope
from utils.params import params as p

from convolution_ops import SKPConv_weighting, KPConv_weighting, resnetb_block_feats
from base_model import Model, FEATS_COMBI_BLOCK


p.define("kp_attention", False)
p.define("kp_attention_heads", [8, 8, 8, 8, 8, 8])
p.define("kp_extent", 1.0)
p.define("num_kpoints", 15)


class SKP_KPConv(Model):
    def __init__(self, local_feats, feats_combi, pooling,
                 bn_decay=None):
        super(SKP_KPConv, self).__init__(local_feats=local_feats,
                                         feats_combi=feats_combi,
                                         pooling=pooling,
                                         bn_decay=bn_decay)

        self.neighbor_indices = tf.placeholder(tf.int32,
                                               [None,
                                                p.neigh_nb],
                                               name="neighbor_indices")

        # --- Model properties ------------------------------------------------
        self.inference
        self.loss

    def get_feed_dict(self, x_batch, y_batch, is_training):
        feed_dict = self.get_base_feed_dict(x_batch, y_batch, is_training)
        xb_neigh_indices = np.array([x_i[1] for x_i in x_batch])

        for i in range(len(xb_neigh_indices)):
            # Re-number the neighbors based on position in the batch
            mask = xb_neigh_indices[i] != -1
            xb_neigh_indices[i][mask] += i*p.max_support_point

            # Disconnect neighbors with a prob in [0, p.neigh_disconnect)
            if is_training:
                # Keep the self connection
                mask_self_connect = np.array(
                    [[False] + [True]*(p.neigh_nb-1)]*p.max_support_point)

                mask2 = np.logical_and(mask, mask_self_connect)
                disco_rate = p.neigh_disconnect * np.random.rand()
                pt_idx, n_idx = np.nonzero(mask2)
                mask_disc = np.zeros(pt_idx.shape, dtype=int)
                mask_disc[:int(disco_rate*pt_idx.shape[0])] = 1
                np.random.shuffle(mask_disc)
                mask_disc = mask_disc.astype(bool)
                xb_neigh_indices[i][pt_idx[mask_disc], n_idx[mask_disc]] = -1

        xb_neigh_indices = xb_neigh_indices.reshape((-1, p.neigh_nb))

        # Replace -1 index with last element+1 (one element added in (S)KPConv)
        mask = xb_neigh_indices == -1
        xb_neigh_indices[mask] = xb_neigh_indices.shape[0]

        feed_dict[self.neighbor_indices] = xb_neigh_indices

        return feed_dict

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """

        # --- Local feats -----------------------------------------------------
        local_feats = self.get_local_feats()

        # --- Feats Combination -----------------------------------------------
        feats_combi = local_feats
        print "feats_combi/IN:",  feats_combi.get_shape()

        with tf.variable_scope('feats_combi'):
            if self.feats_combi_block == FEATS_COMBI_BLOCK.SKPConv:
                with tf.variable_scope('skpconv_weights'):
                    all_weights = SKPConv_weighting(
                        self.support_pts,
                        self.support_pts,
                        self.neighbor_indices,
                        self.lrf_transforms,
                        num_kpoints=p.num_kpoints,
                        fixed='center',
                        KP_influence='linear',
                        aggregation_mode='sum')
            elif self.feats_combi_block == FEATS_COMBI_BLOCK.KPConv:
                with tf.variable_scope('kpconv_weights'):
                    all_weights = KPConv_weighting(
                        self.support_pts,
                        self.support_pts,
                        self.neighbor_indices,
                        self.lrf_transforms,
                        num_kpoints=p.num_kpoints,
                        fixed='center',
                        KP_extent=p.kp_extent,
                        KP_influence='linear',
                        aggregation_mode='sum')

            for i in range(len(p.feats_combi_layers)):
                with tf.variable_scope('skp_kpconv_' + str(i)):
                    feats_combi = resnetb_block_feats(
                        all_weights,
                        self.neighbor_indices,
                        feats_combi,
                        p.feats_combi_layers[i],
                        self.is_training,
                        num_kernel_points=p.num_kpoints,
                        use_batch_norm=p.use_batch_norm,
                        batch_norm_momentum=p.batch_norm_momentum,
                        reg_constant=p.reg_constant,
                        use_attention=p.kp_attention,
                        attention_heads=p.kp_attention_heads[i])

        print "feats_combi/OUT:",  feats_combi.get_shape()

        # --- Pooling ---------------------------------------------------------
        return self.get_pooling(feats_combi)

    @define_scope
    def loss(self):
        return self.get_base_loss()

