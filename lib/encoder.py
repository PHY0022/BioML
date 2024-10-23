from Bio import SeqIO
import pandas as pd
import numpy as np
import os

AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

class Encoder:
    def __init__(self, posData, negData):
        self.posData = posData
        self.negData = negData

    def ToAAC(self):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

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

        return posAAC, negAAC
    
    def ToPWM(self):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

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
    
    def ToOneHot(self):
        posRecords = [str(record.seq) for record in SeqIO.parse(self.posData, "fasta")]
        negRecords = [str(record.seq) for record in SeqIO.parse(self.negData, "fasta")]

        posOneHot = []
        negOneHot = []

        for record in posRecords:
            oneHot = np.zeros((len(record), len(AAs)), dtype=int)
            for i, aa in enumerate(record):
                oneHot[i][AAs.index(aa)] = 1
            posOneHot.append(oneHot)
        for record in negRecords:
            oneHot = np.zeros((len(record), len(AAs)), dtype=int)
            for i, aa in enumerate(record):
                oneHot[i][AAs.index(aa)] = 1
            negOneHot.append(oneHot)

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