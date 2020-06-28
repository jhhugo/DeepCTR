import pandas as pd
import pickle

from deepctr.models import MMOE
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.metrics import AUC

if __name__ == "__main__":
    train = pd.read_csv(r"D:\Users\hao.guo\deepctr\data\census-income\train.csv")
    valid = pd.read_csv(r"D:\Users\hao.guo\deepctr\data\census-income\valid.csv")
    test = pd.read_csv(r"D:\Users\hao.guo\deepctr\data\census-income\test.csv")

    categorical_columns = {"class_worker": 9, "det_ind_code": 52, "det_occ_code": 47, "education": 17, "hs_college": 3,
                       "major_ind_code": 24, "major_occ_code": 15, "race": 5, "hisp_origin": 10,
                       "union_member": 3, "unemp_reason": 6, "full_or_part_emp": 8, "tax_filer_stat": 6, "region_prev_res": 6,
                       "state_prev_res": 51, "det_hh_fam_stat": 38, "det_hh_summ": 8, "mig_chg_msa": 10, "mig_chg_reg": 9,
                       "mig_move_reg": 10, "mig_same": 3, "mig_prev_sunbelt": 4, "fam_under_18": 5, "country_father": 43,
                       "country_mother": 43, "country_self": 43, "citizenship": 5, "vet_question": 3, "own_or_self": 3,
                       "vet_question": 3, "vet_benefits": 3}

    num_columns = ["age", "wage_per_hour", "capital_gains", "capital_losses", "stock_dividends", "num_emp", "weeks_worked", "sex", "year"]

    for c in num_columns:
        train[c] = train[c].astype("float")
        valid[c] = valid[c].astype("float")   
        test[c] = test[c].astype("float")                      

    sparse_feature_columns = [SparseFeat(c, vocabulary_size=categorical_columns[c], embedding_dim=4, dtype='int32') for c in categorical_columns]

    num_feature_columns = [DenseFeat(c,) for c in num_columns]

    dnn_feature_columns = sparse_feature_columns + num_feature_columns

    feature_names = get_feature_names(dnn_feature_columns)

    train_inputs = {name: train[name].values for name in feature_names}
    train_labels = {"income": train["income"].values, "marital": train["marital"].values}

    valid_inputs = {name: valid[name].values for name in feature_names}
    valid_labels = {"income": valid["income"].values, "marital": valid["marital"].values}

    test_inputs = {name: test[name].values for name in feature_names}
    test_labels = {"income": test["income"].values, "marital": test["marital"].values}


    # 4.Define Model,train,predict and evaluate
    model = MMOE(dnn_feature_columns, labels_dict=["income", "marital"], task=["binary", "binary"], l2_reg_embedding=0)
    loss = {"income": "binary_crossentropy", "marital": "binary_crossentropy"}
    model.compile(optimizer="adam", loss=loss,
                  metrics=[AUC()])

    history = model.fit(train_inputs, train_labels,
                        batch_size=1024, epochs=10, verbose=1, validation_data=(valid_inputs, valid_labels))