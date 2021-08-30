from functools import partial
from utils.params import params as p

from base_model import LOCAL_FEATS_BLOCK, FEATS_COMBI_BLOCK, POOLING_BLOCK
from GAT import GAT
from SKP_KPConv import SKP_KPConv
from EdgeConv import EdgeConv


# # Dropout prob params
p.define("batch_norm_momentum", 0.9)
p.define("use_batch_norm", False)
p.define("pool_drop_prob", 0.5)
p.define("reg_constant", 0.01)

# Model arch params
p.define("feats_combi_layers", [64, 64, 128, 128, 256, 256])
p.define("attention_heads", [8, 8, 8, 8, 8, 8])
p.define("local_feats_layers", [16, 16, 32, 256])

p.define("with_height", False)
p.define("with_rel_scales", False)
p.define("with_rescaled_pos", False)
p.define("num_cluster", 1)
p.define("group_radius", 0.1)
p.define("with_bbox", False)
p.define("vote_loss_weight", 1.)

p.define("neigh_disconnect", 0.)


def get_model(local_feats, feats_combi, pooling):
    if type(local_feats) == str:
        try:
            local_feats = LOCAL_FEATS_BLOCK[local_feats]
        except KeyError:
            raise Exception("Unknown local feats block ! Check the name again")

    if type(feats_combi) == str:
        try:
            feats_combi = FEATS_COMBI_BLOCK[feats_combi]
        except KeyError:
            raise Exception("Unknown feats combi block ! Check the name again")

    if type(pooling) == str:
        try:
            pooling = POOLING_BLOCK[pooling]
        except KeyError:
            raise Exception("Unknown pooling block ! Check the name again")

    if feats_combi in [FEATS_COMBI_BLOCK.KPConv, FEATS_COMBI_BLOCK.SKPConv]:
        return partial(SKP_KPConv,
                       local_feats=local_feats,
                       feats_combi=feats_combi,
                       pooling=pooling)

    if feats_combi == FEATS_COMBI_BLOCK.GAT:
        return partial(GAT,
                       local_feats=local_feats,
                       feats_combi=feats_combi,
                       pooling=pooling)

    if feats_combi == FEATS_COMBI_BLOCK.EdgeConv:
        return partial(EdgeConv,
                       local_feats=local_feats,
                       feats_combi=feats_combi,
                       pooling=pooling)
