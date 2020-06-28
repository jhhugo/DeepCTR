from itertools import chain
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input
from ..layers.core import PredictionLayer, DNN, SampledSoftmax
from ..layers.utils import concat_func, add_func, basic_loss_function
from ..layers.interaction import Weighted_Expert_Network


def MMOE(dnn_feature_columns, labels_dict, expert_nums=8, task_nums=2, expert_hidden_units=16, dnn_hidden_units=(8, ), init_std=0.0001, 
         expert_activation="relu", gate_activation="softmax", use_bias=True, use_omoe=False, l2_reg_embedding=0.00001, l2_reg_dnn=0,
         seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task=['binary', "binary"]
        ):
    features = build_input_features(dnn_feature_columns)

    group_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, seq_mask_zero=True, support_dense=True, support_group=False)

    dnn_input = combined_dnn_input(group_embedding_list, dense_value_list)
    
    # Weighted_Expert_Network
    weighted_expert_outputs = Weighted_Expert_Network(expert_nums, task_nums, use_bias=use_bias, use_omoe=use_omoe, l2_reg=l2_reg_dnn, seed=seed)(dnn_input)

    outputs = []
    # 每个task的tower
    for idx in range(task_nums):
        if not use_omoe:
            dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                            dnn_use_bn, seed)(weighted_expert_outputs[idx])
        else:
            dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                            dnn_use_bn, seed)(weighted_expert_outputs)

        dnn_logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(dnn_output)

        output = PredictionLayer(task[idx], name=labels_dict[idx])(dnn_logit)
        outputs.append(output)
    model = Model(inputs=features, outputs=outputs)

    return model
