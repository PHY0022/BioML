from Bio import SeqIO
import pandas as pd
import numpy as np
import os
from itertools import product

AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def GetLebel(posData, negData):
    return posData + negData, [1] * len(posData) + [0] * len(negData)

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

def Balance(posRecords, negRecords, upsample=False):
    posSize = len(posRecords)
    negSize = len(negRecords)
    if posSize > negSize:
        if upsample:
            posRecords, negRecords = UpSample(posRecords, negRecords)
        else:
            posRecords, negRecords = DownSample(posRecords, negRecords)
    elif posSize < negSize:
        if upsample:
            negRecords, posRecords = UpSample(negRecords, posRecords)
        else:
            negRecords, posRecords = DownSample(negRecords, posRecords)
    return posRecords, negRecords

def DownSample(larger, smaller):
    if len(larger) < len(smaller):
        ValueError
    size = len(smaller)
    # Shuffle larger
    np.random.shuffle(larger)
    return larger[:size], smaller

def UpSample(larger, smaller):
    if len(larger) < len(smaller):
        ValueError
    size = len(larger)
    # Shuffle smaller
    np.random.shuffle(smaller)
    smaller = smaller + smaller[:size - len(smaller)]
    return larger, smaller
    

class Encoder:
    def __init__(self, posData, negData, balance=False, upsample=False):
        self.posData = posData
        self.negData = negData
        self.balance = balance
        self.upsample = upsample

    def ToSeq(self):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        if self.balance:
            posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)

        return posRecords, negRecords
    
    def ToKmer(self, k):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        if self.balance:
            posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)

        posKmers = []
        negKmers = []

        for record in posRecords:
            Kmers = []
            for i in range(len(record) - k + 1):
                Kmers.append(record[i:i+k])
            posKmers.append(Kmers)
        for record in negRecords:
            Kmers = []
            for i in range(len(record) - k + 1):
                Kmers.append(record[i:i+k])
            negKmers.append(Kmers)

        del posRecords, negRecords

        return posKmers, negKmers

    def ToAAC(self):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        if self.balance:
            posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)

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
    
    def ToPWM(self):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        if self.balance:
            posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)

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
    
    def ToOneHot(self, remove_center=False):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        if self.balance:
            posRecords, negRecords = Balance(posRecords, negRecords, self.upsample)

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