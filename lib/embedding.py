from . import models, encoder
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np

def skipgrams_kmer(Kmers, window_size):
    pairs = []
    for i, Kmer in enumerate(Kmers):
        l = i - window_size
        r = i + window_size + 1
        if l < 0:
            l = 0
        if r > len(Kmers):
            r = len(Kmers)
        for j in range(l, r):
            if i == j:
                continue
            pairs.append((Kmer, Kmers[j]))
    return pairs

def skip_gram_word2vec(data, input_dim, embedding_dim=200, window_size=2, epochs=500, batch_size=32):
    # Generate Skip-Gram pairs
    pairs = []
    # lebel = []
    for Kmers in data:
        # pair, label = skipgrams(seq, vocabulary_size=input_dim, window_size=window_size, negative_samples=0)
        new_pairs = skipgrams_kmer(Kmers, window_size)
        pairs.extend(new_pairs)
    # print(pairs[:10])

    # Turn AA into one-hot encoding
    kmerEncoder = encoder.KmerEncoder(len(pairs[0][0]))
    x_train = []
    y_train = []
    for pair in pairs:
        x_train.append(kmerEncoder.Kmer2OneHot(pair[0]))
        y_train.append(kmerEncoder.Kmer2OneHot(pair[1]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape, y_train.shape)

    # Build Word2Vec model
    word2vec = models.Word2Vec(input_dim, embedding_dim)
    word2vec.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # input = [kmerEncoder.Kmer2OneHot("AA")]
    # input = np.array(input)
    # print(input.shape)
    # print(np.array(word2vec.predict(input)).shape)
    print(word2vec.model.summary())

    return word2vec