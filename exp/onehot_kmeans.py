import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath( os.path.join(current_dir, "../") )
sys.path.append(parent_dir)

from lib import encoder, utils
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import time


time_start = time.time()


pos_path = "dataset/31mer/provided_by_TA/positive_clustered_sequences.fasta"
neg_path = "dataset/31mer/provided_by_TA/negative_clustered_sequences.fasta"
# pos_path = "dataset/31mer/test/positive_cd_hit50.test.fasta"
# neg_path = "dataset/31mer/test/negative_cd_hit50.test.fasta"


Seqs = encoder.Encoder(pos_path, neg_path)


posOnehot, negOnehot = Seqs.ToOneHot()
# posOnehot = posOnehot[:10]
print(len(posOnehot), len(negOnehot))
# 


posOnehot, negOnehot = np.array(posOnehot), np.array(negOnehot)
# dim1, dim2, dim3 = posOnehot.shape


# Reshape the data to 2 dim
posOnehot = np.reshape(posOnehot, (posOnehot.shape[0], -1))
negOnehot = np.reshape(negOnehot, (negOnehot.shape[0], -1))
print(posOnehot.shape, negOnehot.shape)


# K-Means
clusered_indices = encoder.k_means_indices(negOnehot, posOnehot)
print(len(posOnehot))
print(len(negOnehot[clusered_indices]))
# print(len(clustered_neg_seq))


save_path = "dataset/31mer/provided_by_TA/kmeans/"
np.save(save_path+"onehot_kmeans_indices.npy", clusered_indices)


print(f'Time elapsed:', utils.format_time(time.time() - time_start))


# Save the clustered negative sequences
# save_path = "dataset/31mer/provided_by_TA/kmeans/"
# os.makedirs(save_path, exist_ok=True)
# with open(save_path+"negative_onehot_kmeans_sequences.fasta", "w") as f:
#     for i, seq in enumerate(clustered_neg_seq):
#         f.write(f">{i+1}\n{seq}\n")