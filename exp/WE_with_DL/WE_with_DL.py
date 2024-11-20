'''
#####################################
# Word Embedding with Deep Learning #
#####################################

Implemented based on the paper:

"Incorporating Deep Learning With Word Embedding to Identify Plant Ubiquitylation Sites"
link: https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2020.572195/full
'''
import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath( os.path.join(current_dir, "../../") )
sys.path.append(grandparent_dir)

from lib import encoder, models, embedding, evaluate
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU detected. Use CPU? (y/n)")
    using_cpu = input()
    if using_cpu.lower() != 'y':
        raise RuntimeError("No GPU detected. Exiting...")


##### Hyperparameters #####
word2vec_epochs = 20
word2vec_batch_size = 64

DL_epochs = 120
DL_batch_size = 128

train_test_split_test_size = 0.2

pretrained = True#False
save_model = True
word2vec_model_path = os.path.join(current_dir, "pretrained/word2vec.model")
DL_model_path = os.path.join(current_dir, "pretrained/WE_DL.model")
###########################


# Read in data
# Seqs = encoder.Encoder("dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta",
#                        "dataset/31mer/provided_by_TA/negative_clustered_sequences.fasta",
#                        balance=True, upsample=True)
Seqs = encoder.Encoder("dataset/31mer/test/positive_cd_hit50.test.fasta",
                       "dataset/31mer/test/negative_cd_hit50.test.fasta",
                       balance=True, upsample=True)

# Convert to kmers
k = 2
print("Converting to kmers...")
posKmers, negKmers = Seqs.ToKmer(k)
data = posKmers + negKmers
X_data = data
y_data = [1] * len(posKmers) + [0] * len(negKmers)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=train_test_split_test_size, random_state=87)

# Train word2vec
if pretrained:
    print("Loading pretrained word2vec model...")
    word2vec = models.Word2Vec(0, 0)
    word2vec.model = load_model(word2vec_model_path)
else:
    print("Training word2vec...")
    word2vec = embedding.skip_gram_word2vec(X_train, len(encoder.AAs) ** k, epochs=word2vec_epochs, batch_size=word2vec_batch_size)
    if save_model:
        word2vec.save(word2vec_model_path)

# Convert to one-hot encoding
print("Converting to one-hot encoding...")
kmerEncoder = encoder.KmerEncoder(k)
X_train_onehot = []
for kmers in X_train:
    oneHot = []
    for kmer in kmers:
        oneHot.append(kmerEncoder.Kmer2OneHot(kmer))
    X_train_onehot.append(oneHot)
X_train = np.array(X_train_onehot)
y_train = np.array(y_train)
del X_train_onehot

X_test_onehot = []
for kmers in X_test:
    oneHot = []
    for kmer in kmers:
        oneHot.append(kmerEncoder.Kmer2OneHot(kmer))
    X_test_onehot.append(oneHot)
X_test = np.array(X_test_onehot)
y_test = np.array(y_test)
del X_test_onehot

# Train WE_DL
we_dl = models.WE_DL(word2vec)
if pretrained:
    print("Loading pretrained model...")
    # we_dl = models.WE_DL(None)
    # we_dl.load(DL_model_path, word2vec)
    # we_dl = models.WE_DL(word2vec)
    we_dl.model = load_model(DL_model_path)
    # we_dl.word2vec = word2vec
else:
    print("Training WE_DL...")
    we_dl.fit(X_train, y_train, epochs=DL_epochs, batch_size=DL_batch_size)
    we_dl.model.summary()
    if save_model:
        we_dl.save(DL_model_path)
# raise RuntimeError

# Evaluate
print("========== Evaluation ==========")
# Predict
y_pred = we_dl.predict(X_test)

# ROC curve
evaluate.ROC_curve(y_test, y_pred, current_dir)
# tprs, fprs, thres = metrics.roc_curve(y_test, y_pred)
# auc = metrics.auc(fprs, tprs)
# plt.plot(fprs, tprs, label=f"WE_DL (AUC = {auc:.2f})")
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

# best_thres = thres[np.argmax(tprs - fprs)]

# # Classification report
# y_pred = [1 if pred > best_thres else 0 for pred in y_pred]
# accuracy = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_test)]) / len(y_test)
# print(f"Accuracy: {accuracy}")
# print("Classification report:")
# print(metrics.classification_report(y_test, y_pred))
# print("Confusion matrix:")
# print(metrics.confusion_matrix(y_test, y_pred))