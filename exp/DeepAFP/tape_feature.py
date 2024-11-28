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

posSeqs, negSeqs = Seqs.ToSeq()


posTAPE, negTAPE = embedding.tape_embedding(posSeqs, negSeqs)


print(posTAPE.shape, negTAPE.shape)


prefix = "train_"
np.save(current_dir+"/data/"+prefix+"posTAPE.npy", posTAPE)
np.save(current_dir+"/data/"+prefix+"negTAPE.npy", negTAPE)