# -*- coding:utf-8 -*-
"""
    Author: gh
"""

from itertools import chain
import tensorflow as tf

from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input, DEFAULT_GROUP_NAME
from ..layers.core import PredictionLayer
from ..layers.interaction import ResNet
from ..layers.utils import concat_func, add_func

def DeepCrossing(dnn_feature_columns, dnn_hidden_units=(128, ), l2_reg_embedding=0.00001, l2_reg_dnn=0, 
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    # 生成feature input 字典
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, support_group=False)
    
    # stack
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = ResNet(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        dnn_use_bn, seed)(dnn_input)

    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, 
                                      activation=None)(dnn_output)
        
    output = PredictionLayer(task)(dnn_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model