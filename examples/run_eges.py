import pandas as pd
import pickle

from deepctr.models import EGES
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
from deepctr.layers.utils import basic_loss_function

if __name__ == "__main__":
    data = pd.read_csv('./data_cache/all_data.csv')
    with open("./data_cache/count.pkl", "rb") as f:
        count_dict = pickle.load(f)

    item_feature = SparseFeat("item_id", vocabulary_size=count_dict["item_id"], embedding_dim=128, dtype='int32')

    side_features = ["brand_id", "shop_id", "cate_id"]
    target = ['label']

    side_feature_columns = [SparseFeat(feat, vocabulary_size=count_dict[feat], embedding_dim=128, dtype='int32')
                              for feat in side_features]

    feature_names = get_feature_names([item_feature] + side_feature_columns)


    model_inputs = {name: data[name].values for name in feature_names}
    model_inputs["label"] = data["label"].values


    # 4.Define Model,train,predict and evaluate
    model = EGES(item_feature, side_feature_columns, num_sampled=100, l2_reg_embedding=0)
    model.compile(optimizer="adam", loss=basic_loss_function,
                  metrics=None)

    history = model.fit(model_inputs, data[target].values,
                        batch_size=2048, epochs=2, verbose=2, validation_split=0.2)