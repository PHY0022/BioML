'''
####################
# WE_DL_v1 Trainer #
####################

Code for training WE_DL_v1 model.
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



def kmers2onehot(kmers, k=2):
    kmerEncoder = encoder.KmerEncoder(k)
    X_data_onehot = []
    for kmers in kmers:
        oneHot = []
        for kmer in kmers:
            oneHot.append(kmerEncoder.Kmer2OneHot(kmer))
        X_data_onehot.append(oneHot)
    return np.array(X_data_onehot)



def trainer(X_kmer,
            y_data,
            fold=5,
            word2vec_epochs=15,
            word2vec_batch_size=256,
            epochs=100,
            batch_size=128):
    # params = {}
    time = CurrentTime()
    print("Time:", time)

    kf = KFold(n_splits=fold, shuffle=True)

    result_dir = os.path.join(current_dir, "result-"+time)
    os.makedirs(result_dir, exist_ok=True)

    open(os.path.join(result_dir, "params.txt"), "w").write(f"time={time}\nword2vec_epochs={word2vec_epochs}\nword2vec_batch_size={word2vec_batch_size}\nepochs={epochs}\nbatch_size={batch_size}\n")

    print("Cross validation...")
    with open(os.path.join(result_dir, "result.csv"), "w") as fw:
        with open(os.path.join(result_dir, "raw.csv"), "w") as fw1:
            fw.write("fold,acc,tn, fp, fn, tp,sensitivity,specificity,mcc,auc\n")
            for i, (train_idxs, test_idxs) in enumerate(kf.split(X_kmer)):
                print(f"Fold {i+1}...")

                # X_kmer_train, X_data_test = X_kmer[train_idxs], X_kmer[test_idxs]
                X_kmer_train, X_data_test = [X_kmer[i] for i in train_idxs], [X_kmer[i] for i in test_idxs]
                # y_data_train, y_data_test = y_data[train_idxs], y_data[test_idxs]
                y_data_train, y_data_test = [y_data[i] for i in train_idxs], [y_data[i] for i in test_idxs]
                y_data_train = np.array(y_data_train)
                y_data_test = np.array(y_data_test)

                print("Training word2vec...")
                word2vec = embedding.skip_gram_word2vec(X_kmer_train, len(encoder.AAs) ** 2, epochs=word2vec_epochs, batch_size=word2vec_batch_size)

                # Convert kmers to one-hot encoding
                print("Converting kmers to one-hot encoding...")
                X_train = kmers2onehot(X_kmer_train)

                # Train model
                model = models.WE_DL(word2vec)
                model.fit(X_train, y_data_train, epochs=epochs, batch_size=batch_size)

                # Test model
                X_test = kmers2onehot(X_data_test)

                y_proba = model.predict(X_test)[: ,0]

                y_proba = np.array(y_proba, dtype=float)
                acc, tn, fp, fn, tp, sn, sp, mcc, auc = evaluate.Result(y_data_test[:, 0], y_proba)
                print(f"Fold {i+1}: acc={acc:.4f}, tn={tn}, fp={fp}, fn={fn}, tp={tp}, sn={sn:.4f}, sp={sp:.4f}, mcc={mcc:.4f}, auc={auc:.4f}")

                # Save result
                fw.write(f"{i+1},{acc},{tn},{fp},{fn},{tp},{sn},{sp},{mcc},{auc}\n")
                fw1.write(f"{i+1},")
                for i in range(len(y_proba)):
                    fw1.write(f"{y_data_test[i]},")
                fw1.write(f"\n{i+1},")
                for proba in y_proba:
                    fw1.write(f"{proba},")
                fw1.write("\n")

                del model


    print("Training word2vec...")
    word2vec = embedding.skip_gram_word2vec(X_kmer, len(encoder.AAs) ** 2, epochs=word2vec_epochs, batch_size=word2vec_batch_size)

    # Convert kmers to one-hot encoding
    print("Converting kmers to one-hot encoding...")
    X_train = kmers2onehot(X_kmer)

    # Train model
    print("Training model...")
    model = models.WE_DL(word2vec)
    model.fit(X_train, y_data, epochs=epochs, batch_size=batch_size)

    word2vec_path = os.path.join(result_dir, "word2vec.model")
    word2vec.save(word2vec_path)
    model_path = os.path.join(result_dir, "WE_DL.model")
    model.save(model_path)
    print("Models saved at", result_dir)



def main():
    # Check for GPU
    models.check_gpu()


    # Load data
    pos_path = "dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta"
    neg_path = "dataset/31mer/provided_by_TA/negative_clustered_sequences.fasta"
    # pos_path = "dataset/31mer/test/positive_cd_hit50.test.fasta"
    # neg_path = "dataset/31mer/test/negative_cd_hit50.test.fasta"

    Seqs = encoder.Encoder(pos_path, neg_path)

    print("Converting sequences to kmers...")
    k = 2
    posKmers, negKmers = Seqs.ToKmer(k)

    posKmers, negKmers = encoder.Balance(posKmers, negKmers, shuffle=False)

    X_kmer, y_data = encoder.GetLebel(posKmers, negKmers)
    y_data = encoder.OneHot2Label(y_data)


    # Model training
    trainer(X_kmer, y_data, word2vec_epochs=20, epochs=200)


    print("[ok]")



if __name__ == "__main__":
    main()