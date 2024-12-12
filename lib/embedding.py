from . import models, encoder
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np
import torch
from tape import ProteinBertModel, TAPETokenizer
import ankh
import esm

def skipgrams_kmer(Kmers, window_size):
    pairs = []
    if len(Kmers) == 0:
        return pairs
    
    k = len(Kmers[0])
    for i, Kmer in enumerate(Kmers):
        l = (i - window_size) * k
        r = (i + window_size + 1) * k
        ### * k because kmers is converted to k-mer by order
        ### so the non-overlapping window size is k * window_size
        ### e.g. origin seq: YITRNKRKKL, k = 2, window_size = 2
        ### k-mers: YI, IT, TR, RN, NK, KR, RK, KK, KL
        ### non-overlapping pairs for YI(0): TR(2), NK(4)
        ### non-overlapping pairs for TR(2): YI(0), RN(4), KR(6)

        if l < 0:
            l = 0
        if r > len(Kmers):
            r = len(Kmers)
        for j in range(l, r, k):
            if i == j:
                continue
            pairs.append((Kmer, Kmers[j]))

    # print(f"Kmers: {Kmers[:5]}")
    # print(f"Pairs: {pairs[:5]}")
    # exit()
    return pairs

def skip_gram_word2vec(data, input_dim, embedding_dim=200, window_size=2, epochs=500, batch_size=32):
    '''
    Generate Skip-Gram Word2Vec model

    The downstream use of this model is to generate embeddings for k-mers in a sequence

    Input kmers should be the same order as the sequence

    Example:
    seq = 'YITRNKRKKL', k = 2

    kmers = ['YI', 'IT', 'TR', 'RN', 'NK', 'KR', 'RK', 'KK', 'KL']

    data = list of kmers

    input_dim = sequence_length - k + 1
    '''
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

    print(word2vec.model.summary())

    return word2vec

def tape_embedding(posSeqs, negSeqs, pooled=True, group_num=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ProteinBertModel.from_pretrained('bert-base').to(device)
    tokenizer = TAPETokenizer(vocab='iupac')

    posTokens = [tokenizer.encode(seq) for seq in posSeqs]
    negTokens = [tokenizer.encode(seq) for seq in negSeqs]
    del posSeqs, negSeqs

    posTokens = torch.tensor(np.array(posTokens)).to(device)
    negTokens = torch.tensor(np.array(negTokens)).to(device)

    posOutput = []
    negOutput = []

    if len(posTokens) > 0:
        print(f"Total {len(posTokens)} positive sequences")
        for i in range(0, len(posTokens), group_num):
            left = i
            right = i + group_num
            if right > len(posTokens):
                right = len(posTokens)

            print(f"Embedding {left} to {right}...")

            if not pooled:
                tape = model(posTokens[left:right])[0].to('cpu').detach().numpy()
            else:
                tape = model(posTokens[left:right])[1].to('cpu').detach().numpy()
            if i == 0:
                posOutput = tape
            else:
                posOutput = np.concatenate((posOutput, tape), axis=0)
            
            print(posOutput.shape)
        del posTokens

    if len(negTokens) > 0:
        print(f"Total {len(negTokens)} negative sequences")
        for i in range(0, len(negTokens), group_num):
            left = i
            right = i + group_num
            if right > len(negTokens):
                right = len(negTokens)

            print(f"Embedding {left} to {right}...")

            if not pooled:
                tape = model(negTokens[left:right])[0].to('cpu').detach().numpy()
            else:
                tape = model(negTokens[left:right])[1].to('cpu').detach().numpy()
            if i == 0:
                negOutput = tape
            else:
                negOutput = np.concatenate((negOutput, tape), axis=0)

            print(negOutput.shape)
        del negTokens
    # print(len(posOutput), len(posOutput[0]), posOutput[0].detach().numpy().shape)

    return posOutput, negOutput

def ankh_embedding(posSeqs, negSeqs, group_num=500, model_size='base'):
    if model_size.lower() == 'large':
        print('Using Ankh Large model')
        model, tokenizer = ankh.load_large_model()
        model.eval()
    else:
        print('Using Ankh Base model')
        model, tokenizer = ankh.load_base_model()
        model.eval()

    posOutput = []
    negOutput = []

    for i in range(0, len(posSeqs), group_num):
        left = i
        right = i + group_num
        if right > len(posSeqs):
            right = len(posSeqs)

        print(f"Embedding {left} to {right}...")

        protein_sequences = [list(seq) for seq in posSeqs[left:right]]

        tokens = tokenizer.batch_encode_plus(protein_sequences, 
                                    add_special_tokens=True, 
                                    padding=True, 
                                    is_split_into_words=True, 
                                    return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

        outputs = np.mean(outputs.last_hidden_state.detach().numpy(), axis=1) # Average pooling

        if i == 0:
            posOutput = outputs
        else:
            posOutput = np.concatenate((posOutput, outputs), axis=0)

        print(posOutput.shape)

        del tokens, outputs
    del posSeqs

    for i in range(0, len(negSeqs), group_num):
        left = i
        right = i + group_num
        if right > len(negSeqs):
            right = len(negSeqs)

        print(f"Embedding {left} to {right}...")

        protein_sequences = [list(seq) for seq in negSeqs[left:right]]

        tokens = tokenizer.batch_encode_plus(protein_sequences, 
                                    add_special_tokens=True, 
                                    padding=True, 
                                    is_split_into_words=True, 
                                    return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

        outputs = np.mean(outputs.last_hidden_state.detach().numpy(), axis=1) # Average pooling

        if i == 0:
            negOutput = outputs
        else:
            negOutput = np.concatenate((negOutput, outputs), axis=0)

        print(negOutput.shape)

        del tokens, outputs
    del negSeqs

    return posOutput, negOutput


def esm2_embedding(posSeqs, negSeqs, group_num=500):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    posOutput = []
    negOutput = []

    posData = []
    for i, seq in enumerate(posSeqs):
        posData.append((f"{i}", seq))
    del posSeqs
    negData = []
    for i, seq in enumerate(negSeqs):
        negData.append((f"{i}", seq))
    del negSeqs


    ### Positive sequences
    batch_labels, batch_strs, batch_tokens = batch_converter(posData)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


    print(f"Total {len(batch_tokens)} positive sequences")
    for i in range(0, len(batch_tokens), group_num):
        left = i
        right = i + group_num
        if right > len(batch_tokens):
            right = len(batch_tokens)

        print(f"Embedding {left} to {right}...")

        with torch.no_grad():
            results = model(batch_tokens[left:right], repr_layers=[33], return_contacts=True)
        # token_representations.append(results["representations"][33])
        if i == 0:
            token_representations = results["representations"][33]
        else:
            token_representations = torch.cat((token_representations, results["representations"][33]), dim=0)

        del results


    for i, tokens_len in enumerate(batch_lens):
        posOutput.append(token_representations[i, 1 : tokens_len - 1].mean(0).detach().numpy())

    del batch_labels, batch_strs, batch_tokens, batch_lens, token_representations


    ### Negative sequences
    batch_labels, batch_strs, batch_tokens = batch_converter(negData)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


    print(f"Total {len(batch_tokens)} negative sequences")
    for i in range(0, len(batch_tokens), group_num):
        left = i
        right = i + group_num
        if right > len(batch_tokens):
            right = len(batch_tokens)

        print(f"Embedding {left} to {right}...")

        with torch.no_grad():
            results = model(batch_tokens[left:right], repr_layers=[33], return_contacts=True)
        # token_representations.append(results["representations"][33])
        if i == 0:
            token_representations = results["representations"][33]
        else:
            token_representations = torch.cat((token_representations, results["representations"][33]), dim=0)

        del results


    for i, tokens_len in enumerate(batch_lens):
        negOutput.append(token_representations[i, 1 : tokens_len - 1].mean(0).detach().numpy())

    del batch_labels, batch_strs, batch_tokens, batch_lens, token_representations

    posOutput = np.array(posOutput)
    negOutput = np.array(negOutput)

    return posOutput, negOutput