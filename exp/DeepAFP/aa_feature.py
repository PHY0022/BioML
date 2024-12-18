import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath( os.path.join(current_dir, "../../") )
sys.path.append(grandparent_dir)

from lib import encoder, embedding
import numpy as np


pos_path = "dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta"
neg_path = "dataset/31mer/provided_by_TA/negative_clustered_sequences.fasta"
Seqs = encoder.Encoder(pos_path, neg_path)


# Get features
posOneHot, negOneHot = Seqs.ToOneHot()
posBLOSUM62, negBLOSUM62 = Seqs.ToBLOSUM62()
posZScale, negZScale = Seqs.ToZScale()


os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)


posOneHot, negOneHot = np.array(posOneHot), np.array(negOneHot)
np.save(os.path.join(current_dir, "data", "train_posOneHot.npy"), posOneHot)
np.save(os.path.join(current_dir, "data", "train_negOneHot.npy"), negOneHot)

posBLOSUM62, negBLOSUM62 = np.array(posBLOSUM62), np.array(negBLOSUM62)
np.save(os.path.join(current_dir, "data", "train_posBLOSUM62.npy"), posBLOSUM62)
np.save(os.path.join(current_dir, "data", "train_negBLOSUM62.npy"), negBLOSUM62)

posZScale, negZScale = np.array(posZScale), np.array(negZScale)
np.save(os.path.join(current_dir, "data", "train_posZScale.npy"), posZScale)
np.save(os.path.join(current_dir, "data", "train_negZScale.npy"), negZScale)

pos_data = np.concatenate((posOneHot, posBLOSUM62, posZScale), axis=2)
neg_data = np.concatenate((negOneHot, negBLOSUM62, negZScale), axis=2)


print(pos_data.shape, neg_data.shape)


# Save data
prefix = "train_"
np.save(current_dir+"/data/"+prefix+"pos_data.npy", pos_data)
np.save(current_dir+"/data/"+prefix+"neg_data.npy", neg_data)

