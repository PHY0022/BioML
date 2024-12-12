from Bio import SeqIO
import pandas as pd
import numpy as np
import os
from itertools import product
from aaindex import aaindex1
from . import BLOSUM62, z_scales
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def add(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.concatenate((a, b))
    else:
        raise ValueError("Invalid data type, must be list or numpy array")

def GetLebel(posData, negData):
    return add(posData, negData), [1] * len(posData) + [0] * len(negData)

def OneHot2Label(y_lebel):
    onehot = []
    for lebel in y_lebel:
        if lebel == 0:
            onehot.append([1, 0])
        else:
            onehot.append([0, 1])
    return np.array(onehot)

def Seq2OneHot(record, remove_center=False):
    size = len(record)
    if remove_center:
        size -= 1
    oneHot = np.zeros((size, len(AAs)), dtype=int)
    idx = 0
    for i, aa in enumerate(record):
        if remove_center and i == len(record) // 2:
            continue
        oneHot[idx][AAs.index(aa)] = 1
        idx += 1
    return oneHot

def Seqs2OneHot(records, remove_center=False):
    oneHot = []
    if len(records) == 0:
        return oneHot

    size = len(records[0])
    if remove_center:
        size -= 1

    for record in records:
        oneHot.append(np.zeros((size, len(AAs)), dtype=int))
        idx = 0
        for i, aa in enumerate(record):
            if remove_center and i == len(record) // 2:
                continue
            oneHot[-1][idx][AAs.index(aa)] = 1
            idx += 1

    return oneHot

def AA2OneHot(aa):
    if aa not in AAs:
        ValueError
    oneHot = np.zeros(len(AAs), dtype=int)
    oneHot[AAs.index(aa)] = 1
    return oneHot

class KmerEncoder:
    def __init__(self, k):
        self.k = k
        self.AA_combinations = [''.join(i) for i in product(AAs, repeat=k)]
        # print(self.AA_combinations)

    def Kmer2OneHot(self, kmer):
        if kmer not in self.AA_combinations:
            print(f"Invalid kmer: {kmer}")
            ValueError
        oneHot = np.zeros((len(self.AA_combinations)), dtype=int)
        oneHot[self.AA_combinations.index(kmer)] = 1
        return oneHot

def k_means_indices(larger, smaller):
    if len(larger) < len(smaller):
        ValueError
    elif len(larger) == len(smaller):
        return np.full(len(larger), True)

    larger = np.array(larger)
    smaller = np.array(smaller)

    # K-means clustering
    k = len(smaller)  # Desired number of clusters
    kmeans = KMeans(n_clusters=k, random_state=87)
    kmeans.fit(larger)

    # Get cluster indices
    indices = np.full(len(larger), False)
    for centroid in kmeans.cluster_centers_:
        dist = euclidean_distances([centroid], larger)
        idx = np.argmin(dist)
        indices[idx] = True

    return indices

def K_Means(posData, negData, num_threads=8):
    posSize = len(posData)
    negSize = len(negData)

    if posSize == 0 or negSize == 0:
        RuntimeError("Data with zero size can't balance")


    if posSize > negSize:
        larger = posData
        smaller = negData
        pos_is_larger = True
    elif posSize < negSize:
        larger = negData
        smaller = posData
        pos_is_larger = False
    else:
        return posData, negData
    
    # larger = np.array(larger)
    # smaller = np.array(smaller)

    # os.environ["OMP_NUM_THREADS"] = f'{num_threads}'

    # # K-means clustering
    # k = len(smaller)  # Desired number of clusters
    # kmeans = KMeans(n_clusters=k, random_state=0)
    # kmeans.fit(larger)


    # # Get cluster representatives
    # representatives = []
    # for cluster in range(k):
    #     cluster_indices = np.where(kmeans.labels_ == cluster)[0]
    #     cluster_sequences = larger[cluster_indices]
    #     centroid = kmeans.cluster_centers_[cluster]
    #     closest_idx = np.argmin(euclidean_distances(cluster_sequences, [centroid]))
    #     representatives.append(larger[cluster_indices[closest_idx]])

    # representatives = np.array(representatives)

    # if pos_is_larger:
    #     return representatives, smaller
    # else:
    #     return smaller, representatives

    indices = k_means_indices(larger, smaller)

    if pos_is_larger:
        return larger[indices], smaller
    else:
        return smaller, larger[indices]

def Balance(posRecords, negRecords, upsample=False, shuffle=True):
    posSize = len(posRecords)
    negSize = len(negRecords)

    if posSize == 0 or negSize == 0:
        RuntimeError("Data with zero size can't balance")

    if posSize > negSize:
        if upsample:
            posRecords, negRecords = UpSample(posRecords, negRecords, shuffle)
        else:
            posRecords, negRecords = DownSample(posRecords, negRecords, shuffle)
    elif posSize < negSize:
        if upsample:
            negRecords, posRecords = UpSample(negRecords, posRecords, shuffle)
        else:
            negRecords, posRecords = DownSample(negRecords, posRecords, shuffle)
    return posRecords, negRecords

def DownSample(larger, smaller, shuffle=True):
    if len(larger) < len(smaller):
        ValueError
    size = len(smaller)
    # Shuffle larger
    if shuffle:
        np.random.shuffle(larger)
    return larger[:size], smaller

def UpSample(larger, smaller, shuffle=True):
    if len(larger) < len(smaller):
        ValueError
    size = len(larger)
    # Shuffle smaller
    if shuffle:
        np.random.shuffle(smaller)
    smaller_out = smaller.copy()
    while len(smaller_out) < size:
        rest = size - len(smaller_out)
        if rest > len(smaller):
            smaller_out = add(smaller_out, smaller)
        else:
            smaller_out = add(smaller_out, smaller[:rest])
    return larger, smaller_out
    

class Encoder:
    def __init__(self, posData, negData, balance=False, upsample=False):
        self.posData = posData
        self.negData = negData
        self.balance = balance
        self.upsample = upsample

    def ToSeq(self):
        posRecords = []
        negRecords = []
        if os.path.exists(self.posData):
            posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        if os.path.exists(self.negData):
            negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        if self.balance:
            posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)

        return posRecords, negRecords
    
    def ToKmer(self, k):
        '''
        Convert protein sequences to kmers with length k.

        Example:

        Input = 'WNGWGY'

        Output = ['WN', 'NG', 'GW', 'WG', 'GY']
        '''
        posRecords, negRecords = self.ToSeq()

        posKmers = []
        negKmers = []

        for record in posRecords:
            Kmers = []
            # print(record)
            for i in range(len(record) - k + 1):
                Kmers.append(record[i:i+k])
            # print(Kmers)
            # exit()
            posKmers.append(Kmers)
        for record in negRecords:
            Kmers = []
            for i in range(len(record) - k + 1):
                Kmers.append(record[i:i+k])
            negKmers.append(Kmers)

        del posRecords, negRecords

        return posKmers, negKmers

    def ToAAC(self):
        '''
        Amino Acid Composition (AAC) of the whole positive and negative dataset
        '''

        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posAAC = pd.DataFrame(index=AAs, columns=['count', 'freq'], dtype=float)
        negAAC = pd.DataFrame(index=AAs, columns=['count', 'freq'], dtype=float)


        for aa in AAs:
            posAAC.at[aa,'count'] = 0
            posAAC.at[aa,'freq'] = 0
            negAAC.at[aa,'count'] = 0
            negAAC.at[aa,'freq'] = 0

        for record in posRecords:
            for aa in AAs:
                posAAC.at[aa,'count'] += float(record.count(aa))
                
        for record in negRecords:
            for aa in AAs:
                negAAC.at[aa,'count'] += float(record.count(aa))

        posAAC['freq'] = posAAC['count'] / posAAC['count'].sum()
        negAAC['freq'] = negAAC['count'] / negAAC['count'].sum()

        del posRecords, negRecords

        return posAAC, negAAC
    
    def ToRecordAAC(self):
        '''
        Amino Acid Composition (AAC) of each record in the positive and negative dataset
        '''

        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posAAC = []
        negAAC = []

        center = len(posRecords[0]) // 2 # center of the sequence

        for record in posRecords:
            AAC = np.zeros(len(AAs), dtype=float)
            for i, aa in enumerate(record):
                if i == center:
                    continue
                AAC[AAs.index(aa)] += 1
            posAAC.append(AAC / len(record)) # sum to 1
        for record in negRecords:
            AAC = np.zeros(len(AAs), dtype=float)
            for i, aa in enumerate(record):
                if i == center:
                    continue
                AAC[AAs.index(aa)] += 1
            negAAC.append(AAC / len(record))

        del posRecords, negRecords

        return posAAC, negAAC
    
    def ToRecordDPC(self):
        '''
        Dipeptide Composition (DPC) of each record in the positive and negative dataset
        '''

        # Convert to kmers
        k = 2
        posKmers, negKmers = self.ToKmer(k)

        # Encode kmers to form vector with length 20 * 20 = 400
        kmerEncoder = KmerEncoder(k)
        length = len(kmerEncoder.AA_combinations)

        pos_onehot = []
        for kmers in posKmers:
            oneHot = np.zeros(length, dtype=float)
            for kmer in kmers:
                oneHot[kmerEncoder.AA_combinations.index(kmer)] += 1
            pos_onehot.append(oneHot / len(kmers)) # sum to 1
        # pos_onehot = np.array(pos_onehot)
        del posKmers

        neg_onehot = []
        for kmers in negKmers:
            oneHot = np.zeros(length, dtype=float)
            for kmer in kmers:
                oneHot[kmerEncoder.AA_combinations.index(kmer)] += 1
            neg_onehot.append(oneHot / len(kmers))
        # neg_onehot = np.array(neg_onehot)
        del negKmers

        return pos_onehot, neg_onehot

 
    def ToAAindex(self, remove_center=False):
        '''
        Given a list of selected AAindex, return the corresponding values 
        of each record in the positive and negative dataset

        58 selected aaindex from the paper:
        "Benchmarking of protein descriptor sets in proteochemometric 
        modeling (part 1): comparative study of 13 amino acid descriptor sets"
        link: https://link.springer.com/article/10.1186/1758-2946-5-41
        '''
        posRecords, negRecords = self.ToSeq()

        posAAindex = []
        negAAindex = []

        selected_indices = [
            "ARGP820103", "BAEK050101", "BHAR880101", "CASG920101", "CHAM810101", "CHAM820101",
            "CHAM830101", "CHAM830107", "CHAM830108", "CHOP780201", "CHOP780202", "CHOP780203",
            "CIDH920105", "COSI940101", "FASG760101", "FAUJ880102", "FAUJ880103", "FAUJ880104",
            "FAUJ880105", "FAUJ880106", "FAUJ880109", "FAUJ880110", "FAUJ880111", "FAUJ880112",
            "FAUJ880113", "GRAR740102", "JANJ780102", "JANJ780103", "JOND920102", "JUNJ780101",
            "KLEP840101", "KOEP990101", "KOEP990102", "KRIW790101", "KYTJ820101", "LEVM760102",
            "LEVM760103", "LEVM760104", "LEVM760105", "LEVM760106", "LEVM760107", "MITS020101",
            "MONM990201", "NISK800101", "NISK860101", "PONP800101", "PONP930101", "RACS770103",
            "RADA880108", "ROSG850101", "ROSG850102", "ROSM880102", "TAKK010101", "VINM940101",
            "WARP780101", "WOLR810101", "ZHOH040102", "ZHOH040103"
            ]

        center = len(posRecords[0]) // 2 # center of the sequence

        for record in posRecords:
            AAindexs = []
            for i, aa in enumerate(record):
                if remove_center and i == center:
                    continue
                aaindex = []
                for indice in selected_indices:
                    aaindex.append(aaindex1[indice]['values'][aa])
                AAindexs.append(aaindex)
            posAAindex.append(AAindexs)

        del posRecords

        for record in negRecords:
            AAindexs = []
            for i, aa in enumerate(record):
                if remove_center and i == center:
                    continue
                aaindex = []
                for indice in selected_indices:
                    aaindex.append(aaindex1[indice]['values'][aa])
                AAindexs.append(aaindex)
            negAAindex.append(AAindexs)

        del negRecords

        return posAAindex, negAAindex
    
    def ToBLOSUM62(self, remove_center=False):
        '''
        BLOSUM62 encoding of each record in the positive and negative dataset
        '''

        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posBLOSUM62 = []
        negBLOSUM62 = []

        center = len(posRecords[0]) // 2 # center of the sequence

        for record in posRecords:
            blosum62 = []
            for i, aa in enumerate(record):
                if i == center and remove_center:
                    continue
                if aa not in BLOSUM62.BLOSUM62:
                    ValueError
                blosum62.append(BLOSUM62.BLOSUM62[aa])
            posBLOSUM62.append(blosum62)
        del posRecords

        for record in negRecords:
            blosum62 = []
            for i, aa in enumerate(record):
                if i == center and remove_center:
                    continue
                if aa not in BLOSUM62.BLOSUM62:
                    ValueError
                blosum62.append(BLOSUM62.BLOSUM62[aa])
            negBLOSUM62.append(blosum62)
        del negRecords

        return posBLOSUM62, negBLOSUM62

    def ToCKSAAP(self, k):
        '''
        Composition of k-Spaced Amino Acid Pairs (CKSAAP)
        '''

        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posCKSAAP = []
        negCKSAAP = []

        kmerEncoder = KmerEncoder(2)
        length = len(kmerEncoder.AA_combinations)

        limit = len(posRecords[0]) - k
        for record in posRecords:
            CKSAAP = np.zeros(length, dtype=float)
            for i in range(limit):
                pair = record[i] + record[i + k]
                CKSAAP[kmerEncoder.AA_combinations.index(pair)] += 1
            posCKSAAP.append(CKSAAP / len(record))
        del posRecords

        for record in negRecords:
            CKSAAP = np.zeros(length, dtype=float)
            for i in range(limit):
                pair = record[i] + record[i + k]
                CKSAAP[kmerEncoder.AA_combinations.index(pair)] += 1
            negCKSAAP.append(CKSAAP / len(record))
        del negRecords

        return posCKSAAP, negCKSAAP
    
    def ToCKSAAGP(self, k):
        '''
        Composition of k-Spaced Amino Acid Group Pairs (CKSAAGP)

        Implementation following the paper:
        'Identifying Antitubercular Peptides via Deep Forest Architecture with Effective Feature Representation'
        link: https://pubs.acs.org/doi/full/10.1021/acs.analchem.3c04196
        '''

        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posCKSAAGP = []
        negCKSAAGP = []

        # AA groups
        AA_groups = [
            ['A', 'V', 'L', 'I', 'M', 'G'], # Aliphatic
            ['F', 'Y', 'W'], # Aromatic
            ['K', 'R', 'H'], # Positively charged
            ['D', 'E'], # Negatively charged
            ['S', 'T', 'N', 'Q', 'C', 'P'], # Uncharged
        ]
        
        for record in posRecords:
            CKSAAGP = np.zeros(len(AA_groups) ** 2, dtype=float)
            for i in range(len(record) - k):
                # pair = record[i] + record[i + k]
                group1 = -1
                group2 = -1
                for j, group in enumerate(AA_groups):
                    if record[i] in group:
                        group1 = j
                    if record[i + k] in group:
                        group2 = j
                CKSAAGP[group1 * len(AA_groups) + group2] += 1
            posCKSAAGP.append(CKSAAGP / len(record))
        del posRecords

        for record in negRecords:
            CKSAAGP = np.zeros(len(AA_groups) ** 2, dtype=float)
            for i in range(len(record) - k):
                # pair = record[i] + record[i + k]
                group1 = -1
                group2 = -1
                for j, group in enumerate(AA_groups):
                    if record[i] in group:
                        group1 = j
                    if record[i + k] in group:
                        group2 = j
                CKSAAGP[group1 * len(AA_groups) + group2] += 1
            negCKSAAGP.append(CKSAAGP / len(record))
        del negRecords

        return posCKSAAGP, negCKSAAGP

    def ToPWM(self):
        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        half_size = len(posRecords[0]) // 2
        size_range = range(-half_size, half_size + 1)
        posPWM = pd.DataFrame(index=AAs, columns=size_range, dtype=float)
        negPWM = pd.DataFrame(index=AAs, columns=size_range, dtype=float)

        for aa in AAs:
            for i in size_range:
                posPWM.at[aa, i] = 0
            for i in size_range:
                negPWM.at[aa, i] = 0

        for record in posRecords:
            for i, aa in enumerate(record):
                posPWM.at[aa, i - half_size] += 1

        for record in negRecords:
            for i, aa in enumerate(record):
                negPWM.at[aa, i - half_size] += 1

        return posPWM / float(len(posRecords)), negPWM / float(len(negRecords))
    
    def ToZScale(self, remove_center=False):
        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posZScale = []
        negZScale = []

        center = len(posRecords[0]) // 2

        for record in posRecords:
            zscale = []
            for i, aa in enumerate(record):
                if i == center and remove_center:
                    continue
                zscale.append(z_scales.zscale[aa])
            posZScale.append(zscale)
        del posRecords

        for record in negRecords:
            zscale = []
            for i, aa in enumerate(record):
                if i == center and remove_center:
                    continue
                zscale.append(z_scales.zscale[aa])
            negZScale.append(zscale)
        del negRecords

        return posZScale, negZScale
    
    def ToOneHot(self, remove_center=False):
        # posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        # negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        # if self.balance:
        #     posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)
        posRecords, negRecords = self.ToSeq()

        posOneHot = Seqs2OneHot(posRecords, remove_center)
        negOneHot = Seqs2OneHot(negRecords, remove_center)

        # size = len(posRecords[0])
        # if remove_center:
        #     size -= 1

        # for record in posRecords:
        #     oneHot = np.zeros((size, len(AAs)), dtype=int)
        #     idx = 0
        #     for i, aa in enumerate(record):
        #         if remove_center and i == len(record) // 2:
        #             continue
        #         oneHot[idx][AAs.index(aa)] = 1
        #         idx += 1
        #     posOneHot.append(oneHot)
        # for record in negRecords:
        #     oneHot = np.zeros((size, len(AAs)), dtype=int)
        #     idx = 0
        #     for i, aa in enumerate(record):
        #         if remove_center and i == len(record) // 2:
        #             if i != 15: print(i)
        #             continue
        #         oneHot[idx][AAs.index(aa)] = 1
        #         idx += 1
        #     negOneHot.append(oneHot)

        return posOneHot, negOneHot
    
    def ToAAPC(self):
        posTsv = './tmp/posAAPC.tsv'
        negTsv = './tmp/negAAPC.tsv'
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --out {posTsv} --type CKSAAP")
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --out {negTsv} --type CKSAAP")

        posAAPC = pd.read_csv(posTsv, sep='\t')
        negAAPC = pd.read_csv(negTsv, sep='\t')

        os.remove(posTsv)
        os.remove(negTsv)

        return posAAPC, negAAPC
    
    def ToPSSM(self):
        posTsv = './tmp/posPSSM.tsv'
        negTsv = './tmp/negPSSM.tsv'
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --out {posTsv} --type PSSM --path examples/predictedProteinProperty")
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --out {negTsv} --type PSSM --path examples/predictedProteinProperty")

        posPSSM = pd.read_csv(posTsv, sep='\t')
        negPSSM = pd.read_csv(negTsv, sep='\t')

        os.remove(posTsv)
        os.remove(negTsv)

        return posPSSM, negPSSM
    
    def ToDPC(self):
        posTsv = './tmp/posDPC.tsv'
        negTsv = './tmp/negDPC.tsv'
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --out {posTsv} --type DPC")
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --out {negTsv} --type DPC")

        posDPC = pd.read_csv(posTsv, sep='\t')
        negDPC = pd.read_csv(negTsv, sep='\t')

        os.remove(posTsv)
        os.remove(negTsv)

        return posDPC, negDPC
    
    def ToTPC(self):
        posTsv = './tmp/posTPC.tsv'
        negTsv = './tmp/negTPC.tsv'
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --out {posTsv} --type TPC")
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --out {negTsv} --type TPC")

        posTPC = pd.read_csv(posTsv, sep='\t')
        negTPC = pd.read_csv(negTsv, sep='\t')

        os.remove(posTsv)
        os.remove(negTsv)

        return posTPC, negTPC
    
    def ToSASA(self):
        posTsv = './tmp/posSASA.tsv'
        negTsv = './tmp/negSASA.tsv'
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --path ./tmp --type ASA")
        os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --path ./tmp --type ASA")

        posSASA = pd.read_csv(posTsv, sep='\t')
        negSASA = pd.read_csv(negTsv, sep='\t')

        os.remove(posTsv)
        os.remove(negTsv)

        return posSASA, negSASA

    # def ToAAC(self):
    #     posAAC_tsv = './tmp/posAAC.tsv'
    #     negAAC_tsv = './tmp/negAAC.tsv'
    #     os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --out {posAAC_tsv} --type AAC")
    #     os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --out {negAAC_tsv} --type AAC")

    #     posAAC = pd.read_csv(posAAC_tsv, sep='\t')
    #     negAAC = pd.read_csv(negAAC_tsv, sep='\t')

    #     os.remove(posAAC_tsv)
    #     os.remove(negAAC_tsv)

    #     # print(posAAC)

    #     return posAAC, negAAC
    
    # def ToPWM(self):
    #     posPWM_tsv = './tmp/posPWM.tsv'
    #     negPWM_tsv = './tmp/negPWM.tsv'
    #     os.system(f"python ./bin/iFeature/iFeature.py --file {self.posData} --out {posPWM_tsv} --type PWM")
    #     os.system(f"python ./bin/iFeature/iFeature.py --file {self.negData} --out {negPWM_tsv} --type PWM")

    #     posPWM = pd.read_csv(posPWM_tsv, sep='\t')
    #     negPWM = pd.read_csv(negPWM_tsv, sep='\t')

    #     os.remove(posPWM_tsv)
    #     os.remove(negPWM_tsv)

    #     return posPWM, negPWM