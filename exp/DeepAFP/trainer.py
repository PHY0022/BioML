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

from lib import encoder, embedding, models, evaluate
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from datetime import datetime



def CurrentTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")



def trainer(X_tape, X_data, y_data, fold=5, epochs=100, batch_size=128):
    # params = {}
    time = CurrentTime()
    print("Time:", time)

    kf = KFold(n_splits=fold, shuffle=True)

    result_dir = os.path.join(current_dir, "result-"+time)
    os.makedirs(result_dir, exist_ok=True)

    open(os.path.join(result_dir, "params.txt"), "w").write(f"epochs={epochs}\nbatch_size={batch_size}\n")

    print("Cross validation...")
    with open(os.path.join(result_dir, "result.csv"), "w") as fw:
        with open(os.path.join(result_dir, "raw.csv"), "w") as fw1:
            fw.write("fold,acc,tn, fp, fn, tp,sensitivity,specificity,mcc,auc\n")
            for i, (train_idxs, test_idxs) in enumerate(kf.split(X_data)):
                print(f"Fold {i+1}...")

                X_tape_train, X_tape_test = X_tape[train_idxs], X_tape[test_idxs]
                X_data_train, X_data_test = X_data[train_idxs], X_data[test_idxs]
                y_data_train, y_data_test = y_data[train_idxs], y_data[test_idxs]

                model = models.DeepAFP()
                model.fit(X_tape_train, X_data_train, y_data_train, epochs=epochs, batch_size=batch_size)

                y_pred = model.predict(X_tape_test, X_data_test)

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
    model = models.DeepAFP()
    model.fit(X_tape, X_data, y_data, epochs=epochs, batch_size=batch_size)

    model_path = os.path.join(result_dir, "DeepAFP.model")
    model.save(model_path)
    print("Model saved at", model_path)



def main():
    # Check for GPU
    models.check_gpu()


    # Load data
    pos_data_path = current_dir + "/data/train_pos_data.npy"
    neg_data_path = current_dir + "/data/train_neg_data.npy"
    pos_data = np.load(pos_data_path)
    neg_data = np.load(neg_data_path)
    pos_data, neg_data = encoder.Balance(pos_data, neg_data, shuffle=False)

    posTAPE_path = current_dir + "/data/train_posTAPE.npy"
    negTAPE_path = current_dir + "/data/train_negTAPE.npy"
    posTAPE = np.load(posTAPE_path)
    negTAPE = np.load(negTAPE_path)
    posTAPE, negTAPE = encoder.Balance(posTAPE, negTAPE, shuffle=False)

    X_data, y_data = encoder.GetLebel(pos_data, neg_data)
    y_data = encoder.OneHot2Label(y_data)
    X_tape = np.concatenate((posTAPE, negTAPE), axis=0)

    del pos_data, neg_data, posTAPE, negTAPE


    # Model training
    trainer(X_tape, X_data, y_data)


    print("[ok]")



if __name__ == "__main__":
    main()