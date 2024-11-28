'''
###########
# DeepAFP #
###########

Implemented based on the paper:

"DeepAFP: An effective computational framework for identifying antifungal peptides based on deep learning"
link: https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4758?msockid=08ab38e43da369ce130e2cc13c3768bb
'''
import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath( os.path.join(current_dir, "../../") )
sys.path.append(grandparent_dir)

from lib import encoder, models, embedding, evaluate
from sklearn.model_selection import train_test_split
import numpy as np



# Check for GPU
models.check_gpu()



# Hyperparameters
suffix = "_241126"
model_path = os.path.join(current_dir, "pretrained/DeepAFP" + suffix + ".model")



# Load data
data_path = "dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta"
data_path = "dataset/31mer/test/positive_cd_hit50.test.fasta"
Seqs = encoder.Encoder(data_path, "")



# Get features
OneHot, _ = Seqs.ToOneHot()
BLOSUM62, _ = Seqs.ToBLOSUM62()
ZScale, _ = Seqs.ToZScale()

OneHot = np.array(OneHot)
BLOSUM62 = np.array(BLOSUM62)
ZScale = np.array(ZScale)

X_data = np.concatenate((OneHot, BLOSUM62, ZScale), axis=2)

print(X_data.shape)



# Get bert embeddings
seqs, _ = Seqs.ToSeq()
tape, _ = embedding.tape_embedding(seqs, _)

X_tape = np.mean(tape, axis=1)

print(X_tape.shape)



# Load model
print("Loading model...")
model = models.DeepAFP().load(model_path)



# Predict
print("Predicting...")
y_pred = model.predict(X_tape, X_data)
# np.save("./result.npy", y_pred)



print("[ok]")