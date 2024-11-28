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

from lib import encoder, models, embedding
from sklearn.model_selection import train_test_split
import numpy as np



# Check for GPU
models.check_gpu()



# Hyperparameters
word2vec_epochs = 20
word2vec_batch_size = 64

DL_epochs = 120
DL_batch_size = 128

random_state = 87
# test_size = 0.2
validate_size = 0.1

optim = 'adam'#'rmsprop'

suffix = "_241126_test"
word2vec_model_path = os.path.join(current_dir, "pretrained/word2vec" + suffix + ".model")
DL_model_path = os.path.join(current_dir, "pretrained/WE_DL" + suffix + ".model")



# Load data
pos_path = "dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta"
neg_path = "dataset/31mer/provided_by_TA/negative_clustered_sequences.fasta"
# pos_path = "dataset/31mer/test/positive_cd_hit50.test.fasta"
# neg_path = "dataset/31mer/test/negative_cd_hit50.test.fasta"

Seqs = encoder.Encoder(pos_path, neg_path, balance=True, upsample=True)



# Convert sequences to 2mers
print("Converting sequences to kmers...")
k = 2
posKmers, negKmers = Seqs.ToKmer(k)
X_data, y_data = encoder.GetLebel(posKmers, negKmers)



# Train Word2Vec model
print("Training word2vec...")
word2vec = embedding.skip_gram_word2vec(X_data, len(encoder.AAs) ** k, epochs=word2vec_epochs, batch_size=word2vec_batch_size)



# Convert kmers to one-hot encoding
print("Converting kmers to one-hot encoding...")
kmerEncoder = encoder.KmerEncoder(k)

X_data_onehot = []
for kmers in X_data:
    oneHot = []
    for kmer in kmers:
        oneHot.append(kmerEncoder.Kmer2OneHot(kmer))
    X_data_onehot.append(oneHot)
del X_data
X_train = np.array(X_data_onehot)
y_train = np.array(y_data)
del y_data, X_data_onehot



# Train-validate split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, stratify=y_train, test_size=validate_size, random_state=random_state)



# Train Deep Learning model
we_dl = models.WE_DL(word2vec, optim=optim)

print("Training WE_DL...")
we_dl.fit(X_train, y_train, epochs=DL_epochs, batch_size=DL_batch_size, validation_data=(X_validate, y_validate))
we_dl.model.summary()



# Save model
we_dl.save(DL_model_path)
word2vec.save(word2vec_model_path)
print("Models saved with suffix:", suffix)