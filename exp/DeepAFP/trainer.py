'''
###################
# DeepAFP Trainer #
###################

Code for training DeepAFP model.
Model implementation details can be found in lib/models.py.

This code will do cross validation first,
then train the model with all data.

All results will be saved in a directory named "result-<time>".
Use example in /exp/DeepAFP_results.ipynb to analyze the results.
'''
import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath( os.path.join(current_dir, "../../") )
sys.path.append(grandparent_dir)

from lib import encoder, embedding, models, evaluate, utils
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from datetime import datetime
import time



def CurrentTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")



def trainer(X_bert, X_data, y_data, prefix="", **params):
    time = CurrentTime()
    print("Time:", time)

    kf = KFold(n_splits=params['fold'], shuffle=True, random_state=87)

    result_dir = os.path.join(current_dir, prefix+"result-"+time)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "params.txt"), "w") as f:
        f.write(f"time={time}\n")
        for param in params.items():
            f.write(f"{param[0]}={param[1]}\n")

    embedding_len = X_bert.shape[1]

    print("Cross validation...")
    with open(os.path.join(result_dir, "result.csv"), "w") as fw:
        with open(os.path.join(result_dir, "raw.csv"), "w") as fw1:
            fw.write("fold,acc,tn, fp, fn, tp,sensitivity,specificity,mcc,auc\n")
            for i, (train_idxs, test_idxs) in enumerate(kf.split(X_data)):
                print(f"Fold {i+1}...")

                X_bert_train, X_bert_test = X_bert[train_idxs], X_bert[test_idxs]
                X_data_train, X_data_test = X_data[train_idxs], X_data[test_idxs]
                y_data_train, y_data_test = y_data[train_idxs], y_data[test_idxs]

                model = models.DeepAFP(cnn_dim=params['cnn_dim'], kernel_size=params['kernel_size'], lstm_dim=params['lstm_dim'], embedding_len=embedding_len)
                model.fit(X_bert_train, X_data_train, y_data_train, epochs=params['epochs'], batch_size=params['batch_size'])

                y_pred = model.predict(X_bert_test, X_data_test)

                # print(y_pred.shape, y_pred[:5])
                y_proba = y_pred[:, 0]
                # print(y_proba.shape, y_proba[:5])

                acc, tn, fp, fn, tp, sn, sp, mcc, auc = evaluate.Result(y_data_test[:, 0], y_proba)
                print(f"Fold {i+1}: acc={acc:.4f}, tn={tn}, fp={fp}, fn={fn}, tp={tp}, sn={sn:.4f}, sp={sp:.4f}, mcc={mcc:.4f}, auc={auc:.4f}")

                # Save result
                fw.write(f"{i+1},{acc},{tn},{fp},{fn},{tp},{sn},{sp},{mcc},{auc}\n")
                fw1.write(f"{i+1},")
                for i in range(len(y_proba)):
                    fw1.write(f"{y_data_test[:, 0][i]},")
                fw1.write(f"\n{i+1},")
                for i in range(len(y_proba)):
                    fw1.write(f"{y_proba[i]},")
                fw1.write("\n")

                del model

    
    print("Training model...")
    model = models.DeepAFP(cnn_dim=params['cnn_dim'], kernel_size=params['kernel_size'], lstm_dim=params['lstm_dim'], embedding_len=embedding_len)
    model.fit(X_bert, X_data, y_data, epochs=params['epochs'], batch_size=params['batch_size'])

    model_path = os.path.join(result_dir, "DeepAFP.model")
    model.save(model_path)
    print("Model saved at", model_path)



def main():
    time_start = time.time()


    # Check for GPU
    models.check_gpu()


    #============= Configurations ==============#
    ## Sampling method
    sampling = "upsampling_random"
    # sampling = "downsampling_onehot_kmeans"

    ## BERT encoding
    # bert_model = 'TAPE'
    # bert_model = 'ESM2'
    # bert_model = 'ANKH_base'
    bert_model = 'PROTRANS'

    ## Model hyperparameters
    param_distribution = {
        "filter_num": [32],#64, 
        "kernel_size": [3],#, 4, 5, 6, 7, 8],
        "lstm_num": [20],#, 40, 60]
    }
    # Result prefix
    prefix = '1216_US_PROTRANS-'
    #===========================================#


    # Load data
    pos_data_path = current_dir + "/data/train_pos_data.npy"
    neg_data_path = current_dir + "/data/train_neg_data.npy"
    pos_data = np.load(pos_data_path)
    neg_data = np.load(neg_data_path)


    posBERT_path = current_dir + "/data/train_pos"+bert_model+".npy"
    negBERT_path = current_dir + "/data/train_neg"+bert_model+".npy"
    posBERT = np.load(posBERT_path)
    negBERT = np.load(negBERT_path)

    # Balance data
    if sampling == "upsampling_random":
        pos_data, neg_data = encoder.Balance(pos_data, neg_data, upsample=True, shuffle=False)
        posBERT, negBERT = encoder.Balance(posBERT, negBERT, upsample=True, shuffle=False)
    elif sampling == "downsampling_onehot_kmeans":
        kmeans_indices = np.load("dataset/31mer/provided_by_TA/kmeans/onehot_kmeans_indices.npy")
        if len(neg_data) != len(kmeans_indices):
            raise ValueError("Length of negative data and kmeans indices are not equal.")
        neg_data = neg_data[kmeans_indices]
        negBERT = negBERT[kmeans_indices]
    else:
        raise ValueError("Invalid sampling method.")
    print(pos_data.shape, neg_data.shape)
    print(posBERT.shape, negBERT.shape)


    X_data, y_data = encoder.GetLebel(pos_data, neg_data)
    y_data = encoder.OneHot2Label(y_data)
    X_bert = np.concatenate((posBERT, negBERT), axis=0)

    del pos_data, neg_data, posBERT, negBERT

    print(len(X_data), len(y_data))


    # Model training
    for filter_num in param_distribution["filter_num"]:
        for kernel_size in param_distribution["kernel_size"]:
            for lstm_num in param_distribution["lstm_num"]:
                params = {
                    "sampling": sampling,
                    "bert_model": bert_model,
                    "fold": 5,
                    "epochs": 200,
                    "batch_size": 128,
                    "cnn_dim": filter_num,
                    "kernel_size": kernel_size,
                    "lstm_dim": lstm_num
                }
                trainer(X_bert, X_data, y_data, prefix=prefix, **params)
                models.reset_keras()


    print(f'Time elapsed:', utils.format_time(time.time() - time_start))
    print("[ok]")



if __name__ == "__main__":
    main()