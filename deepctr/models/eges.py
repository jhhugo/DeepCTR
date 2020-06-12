from itertools import chain
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input, DEFAULT_GROUP_NAME
from ..layers.core import PredictionLayer, DNN, SampledSoftmax
from ..layers.utils import concat_func, add_func, basic_loss_function
from ..layers.interaction import Attention_Eges

def EGES(item_feature, side_feature_columns, num_sampled=100, l2_reg_embedding=0.00001, init_std=0.0001, 
         seed=1024):
    features = build_input_features(
        [item_feature] + side_feature_columns)
    labels = Input(shape=(1, ), dtype=tf.int64, name="label")
    # inputs_list = list(features.values()) + [labels]
    features["label"] = labels

    group_embedding_list, dense_value_list = input_from_feature_columns(features, [item_feature] + side_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, seq_mask_zero=False, support_dense=False, support_group=False)
    # concat (batch_size, num_feat, embedding_size)
    concat_embeds = concat_func(group_embedding_list, axis=1)
    
    # attention
    att_embeds = Attention_Eges(item_feature.vocabulary_size, l2_reg_embedding, seed)([features[item_feature.name], concat_embeds])

    # sample_softmax
    loss = SampledSoftmax(item_feature.vocabulary_size, 100, l2_reg_embedding, seed)([att_embeds, features["label"]])

    model = Model(inputs=features, outputs=loss)
    return model
