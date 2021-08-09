import numpy as np
import tensorflow as tf
from utils.tf import define_scope
from utils.params import params as p

from layers import attn_head
from base_model import Model, FEATS_COMBI_BLOCK


class GAT(Model):
    def __init__(self, local_feats, feats_combi, pooling,
                 bn_decay=None):
        super(GAT, self).__init__(local_feats=local_feats,
                                  feats_combi=feats_combi,
                                  pooling=pooling,
                                  bn_decay=bn_decay)

        self.bias_mat = tf.placeholder(tf.float32,
                                       (None,
                                        p.max_support_point,
                                        p.max_support_point),
                                       name="bias_mat")


        if not p.bias_mat:
            raise Exception("GAT model requires bias_mat")

        # --- Model properties ------------------------------------------------
        self.inference
        self.loss

    def get_feed_dict(self, x_batch, y_batch, is_training):
        feed_dict = self.get_base_feed_dict(x_batch, y_batch, is_training)
        xb_bias_mat = [np.array(x_i[1]) for x_i in x_batch]

        if is_training:
            for mat_idx, bias_mat in enumerate(xb_bias_mat):
                i_indices, j_indices = np.nonzero(bias_mat == 0.)
                for idx in range(len(i_indices)):
                    if np.random.rand() > p.neigh_disconnect:
                        continue
                    if i_indices[idx] != j_indices[idx]:
                        xb_bias_mat[mat_idx][i_indices[idx], j_indices[idx]] = -1e9


        feed_dict[self.bias_mat] = xb_bias_mat

        return feed_dict

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """

        # --- Local feats -----------------------------------------------------
        local_feats = self.get_local_feats()
        local_feats = tf.reshape(local_feats, [-1,
                                               p.max_support_point,
                                               p.local_feats_layers[-1]])

        # --- Feats Combination -----------------------------------------------
        feats_combi = local_feats
        print "feats_combi/IN:",  feats_combi.get_shape()

        with tf.variable_scope('graph_layers'):
            for i in range(len(p.feats_combi_layers)):
                gcn_heads = []
                for head_idx in range(p.attention_heads[i]):
                    head = attn_head(feats_combi,
                                     out_sz=p.feats_combi_layers[i] // p.attention_heads[i],
                                     bias_mat=self.bias_mat,
                                     activation=tf.nn.elu,
                                     reg_constant=p.reg_constant,
                                     is_training=self.is_training,
                                     bn_decay=self.bn_decay,
                                     scope="gat_" + str(i) + "/head_" + str(head_idx))
                    gcn_heads.append(head)

                feats_combi = tf.concat(gcn_heads, axis=-1)

        print "feats_combi/OUT:",  feats_combi.get_shape()

        # --- Pooling ---------------------------------------------------------
        return self.get_pooling(feats_combi)

    @define_scope
    def loss(self):
        return self.get_base_loss()

