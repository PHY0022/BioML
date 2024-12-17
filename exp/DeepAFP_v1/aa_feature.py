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
# posOneHot, negOneHot = Seqs.ToOneHot()
# posBLOSUM62, negBLOSUM62 = Seqs.ToBLOSUM62()
# posZScale, negZScale = Seqs.ToZScale()

# posOneHot, negOneHot = np.array(posOneHot), np.array(negOneHot)
# posBLOSUM62, negBLOSUM62 = np.array(posBLOSUM62), np.array(negBLOSUM62)
# posZScale, negZScale = np.array(posZScale), np.array(negZScale)

# pos_data = np.concatenate((posOneHot, posBLOSUM62, posZScale), axis=2)
# neg_data = np.concatenate((negOneHot, negBLOSUM62, negZScale), axis=2)
pos_data, neg_data = Seqs.ToAAindex(remove_center=False)
pos_data, neg_data = np.array(pos_data), np.array(neg_data)

print(pos_data.shape, neg_data.shape)


# Save data
prefix = "train_"
os.makedirs(current_dir+"/data/", exist_ok=True)
np.save(current_dir+"/data/"+prefix+"pos_aaindex31.npy", pos_data)
np.save(current_dir+"/data/"+prefix+"neg_aaindex31.npy", neg_data)

