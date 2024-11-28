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



# Check for GPU
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

test_size = 0.2
validate_size = 0.1

optim = 'adam'#'rmsprop'

pretrained = True
save_model = True
suffix = "_241119"
word2vec_model_path = os.path.join(current_dir, "pretrained/word2vec" + suffix + ".model")
DL_model_path = os.path.join(current_dir, "pretrained/WE_DL" + suffix + ".model")
###########################



# Read in data
Seqs = encoder.Encoder("dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta",
                       "dataset/31mer/provided_by_TA/negative_clustered_sequences.fasta",
                       balance=True, upsample=True)
# Seqs = encoder.Encoder("dataset/31mer/test/positive_cd_hit50.test.fasta",
#                        "dataset/31mer/test/negative_cd_hit50.test.fasta",
#                        balance=True, upsample=True)



# Convert to kmers
k = 2
print("Converting to kmers...")
posKmers, negKmers = Seqs.ToKmer(k)
data = posKmers + negKmers
X_data = data
y_data = [1] * len(posKmers) + [0] * len(negKmers)



# Train-test-validate split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=test_size + validate_size, random_state=87)
X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, stratify=y_test, test_size=validate_size / (test_size + validate_size), random_state=87)



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

X_validate_onehot = []
for kmers in X_validate:
    oneHot = []
    for kmer in kmers:
        oneHot.append(kmerEncoder.Kmer2OneHot(kmer))
    X_validate_onehot.append(oneHot)
X_validate = np.array(X_validate_onehot)
y_validate = np.array(y_validate)
del X_validate_onehot



# Train WE_DL
we_dl = models.WE_DL(word2vec, optim=optim)
if pretrained:
    print("Loading pretrained model...")
    # we_dl = models.WE_DL(None)
    # we_dl.load(DL_model_path, word2vec)
    # we_dl = models.WE_DL(word2vec)
    we_dl.model = load_model(DL_model_path)
    # we_dl.word2vec = word2vec
else:
    print("Training WE_DL...")
    half = len(X_train) // 2
    we_dl.fit(X_train, y_train, epochs=DL_epochs, batch_size=DL_batch_size, validation_data=(X_validate, y_validate))
    # we_dl.fit(X_train[:half], y_train[:half], epochs=DL_epochs, batch_size=DL_batch_size, validation_data=(X_validate, y_validate))
    # we_dl.fit(X_train[half:], y_train[half:], epochs=DL_epochs, batch_size=DL_batch_size, validation_data=(X_validate, y_validate))
    we_dl.model.summary()
    if save_model:
        we_dl.save(DL_model_path)
# raise RuntimeError



# Evaluate
print("========== Evaluation ==========")
# Predict
y_pred = we_dl.predict(X_test)


# ROC curve
tprs, fprs, thresholds = metrics.roc_curve(y_test, 1 - y_pred)
auc = metrics.auc(fprs, tprs)
plt.plot(fprs, tprs, label=f"WE_DL (AUC = {auc:.2f})")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Find the best threshold (closest to (0, 1))
distances = np.sqrt((fprs - 0)**2 + (tprs - 1)**2)
min_distance_index = np.argmin(distances)
best_threshold = thresholds[min_distance_index]
print(f"Best threshold: {best_threshold}")

# Classification report
y_pred = [1 if pred > best_threshold else 0 for pred in y_pred]
accuracy = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_test)]) / len(y_test)
print(f"Accuracy: {accuracy}")
print("Classification report:")
print(metrics.classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))