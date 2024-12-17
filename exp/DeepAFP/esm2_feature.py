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
# pos_path = "dataset/31mer/test/positive_cd_hit50.test.fasta"
# neg_path = "dataset/31mer/test/negative_cd_hit50.test.fasta"
Seqs = encoder.Encoder(pos_path, neg_path)

posSeqs, negSeqs = Seqs.ToSeq()


posESM2, negESM2 = embedding.esm2_embedding(posSeqs, negSeqs)


print(posESM2.shape, negESM2.shape)
# exit()

prefix = "train_"
os.makedirs(current_dir+"/data", exist_ok=True)
np.save(current_dir+"/data/"+prefix+"posESM2.npy", posESM2)
np.save(current_dir+"/data/"+prefix+"negESM2.npy", negESM2)