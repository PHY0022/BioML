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

from lib import encoder, models
from tensorflow.keras.models import load_model
import numpy as np



# Check for GPU
models.check_gpu()



# Hyperparameters
suffix = ""
word2vec_model_path = os.path.join(current_dir, "pretrained/word2vec" + suffix + ".model")
DL_model_path = os.path.join(current_dir, "pretrained/WE_DL" + suffix + ".model")



# Load data
data_path = "dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta"
Seqs = encoder.Encoder(data_path, "")



# Convert sequences to 2mers
print("Converting sequences to kmers...")
k = 2
X_data, _ = Seqs.ToKmer(k)



# Convert kmers to one-hot encoding
print("Converting kmers to one-hot encoding...")
kmerEncoder = encoder.KmerEncoder(k)

X_data_onehot = []
for kmers in X_data:
    oneHot = []
    for kmer in kmers:
        oneHot.append(kmerEncoder.Kmer2OneHot(kmer))
    X_data_onehot.append(oneHot)
X_test = np.array(X_data_onehot)
del X_data, X_data_onehot



# Load models
print("Loading word2vec model...")
word2vec = models.Word2Vec(0, 0)
word2vec.model = load_model(word2vec_model_path)

print("Loading WE_DL model...")
we_dl = models.WE_DL(word2vec)
we_dl.model = load_model(DL_model_path)



# Predict
y_pred = we_dl.predict(X_test)
print(y_pred.shape)
# np.save("./result.npy", y_pred)



print("[ok]")