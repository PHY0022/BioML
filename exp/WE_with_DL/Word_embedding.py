from lib import encoder, models, embedding

Seqs = encoder.Encoder("dataset/31mer/test/positive_cd_hit50.test.fasta", "dataset/31mer/test/negative_cd_hit50.test.fasta")

posOneHot, negOneHot = Seqs.ToOneHot(remove_center=True)

k = 2
posKmers, negKmers = Seqs.ToKmer(k)

data = posKmers + negKmers

print(len(posKmers), len(negKmers), len(data))

word2vec = embedding.skip_gram_word2vec(data, len(encoder.AAs) ** k, epochs=1)