import os
import subprocess
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal
from . import encoder

class HMM:
    '''
    A wrapper for the HMMER docker image
    '''
    def __init__(self, dataPath):
        if dataPath[-1] != '/':
            dataPath += '/'
        self.dataPath = dataPath
        # self.hmmer_docker_cmd = f'docker run -it --rm -v "{dataPath}:/data" hmmer-docker '
        # self.hmmer_docker_args = ['docker', 'run', '-it', '--rm', '-v', f'{dataPath}:/data', 'hmmer-docker']
        self.hmmer_docker_args = ['docker', 'run', '-it', '--rm', '-v', f'{os.getcwd()}:/data', 'hmmer-docker']

    def hmmBuild(self, msafile):
        print(f'Copying {msafile} to {self.dataPath}')
        shutil.copy(msafile, self.dataPath)
        
        msafile = '/data/' + msafile
        hmmfile_out = '/data/' + self.dataPath + os.path.basename(msafile) + '.hmm'

        args = self.hmmer_docker_args.copy()
        args.extend(['hmmbuild', '--wblosum', '--wid', '0.8', hmmfile_out, msafile])

        result = subprocess.run(args, check=True)
        
        # return result.stdout
        return hmmfile_out

    def hmmSearch(self, hmmfile, seqfile, E=1000, domE=1000, incE=1000, incdomE=1000):
        print(f'Copying {seqfile} to {self.dataPath}')
        shutil.copy(seqfile, self.dataPath)

        seqfile = '/data/' + seqfile

        args = self.hmmer_docker_args.copy()
        # args.extend(['hmmsearch', '-E', '1000000', '--domE', '10000', '--incE', '10000', '--incdomE', '10000', hmmfile, seqfile])
        # args.extend(['hmmsearch', '--max', '--domE', f'{domE}', '--incdomE', f'{incdomE}', hmmfile, seqfile])
        args.extend(['hmmsearch', '--max', '-E', f'{E}', '--domE', f'{domE}', '--incE', f'{incE}', '--incdomE', f'{incdomE}', '--domtblout', '/data/hmm.domtblout.out', '--noali', hmmfile, seqfile])

        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    
class Word2Vec:
    '''
    This model is implemented based on the paper:

    "Incorporating Deep Learning With Word Embedding to Identify Plant Ubiquitylation Sites"
    link: https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2020.572195/full

    This model is used to learn embeddings for the WE_DL model
    '''
    def __init__(self, input_dim, embedding_dim):
        self.model = Sequential([
            Dense(embedding_dim, input_dim=input_dim, activation='relu'),  # Learn embeddings
            Dense(input_dim, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def fit(self, x_train, y_train, epochs, batch_size, verbose=1):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def predict(self, data, verbose=0):
        hidden_layer = self.model.get_layer(index=0)
        model = Sequential(hidden_layer)
        return model.predict(data, verbose=verbose)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

class WE_DL:
    '''
    This model is implemented based on the paper:

    "Incorporating Deep Learning With Word Embedding to Identify Plant Ubiquitylation Sites"
    link: https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2020.572195/full

    A pre-trained Word2Vec model is required
    '''
    def __init__(self, word2vec):
        self.word2vec = word2vec
        initializer = RandomNormal(mean=0.0, stddev=0.01)
        self.model = Sequential([
            Conv1D(128, 3, kernel_initializer=initializer, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.5),
            
            Conv1D(128, 3, kernel_initializer=initializer, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.5),
            
            Conv1D(128, 3, kernel_initializer=initializer, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.5),
            
            Flatten(),
            
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=0.9)
        # self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs, batch_size, verbose=1):
        word2vec = self.word2vec
        x_train = [word2vec.predict(x) for x in x_train]
        x_train = np.array(x_train, dtype=float)
        self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def predict(self, x_test, verbose=0):
        word2vec = self.word2vec
        x_test = np.array(x_test, dtype=float)
        # print(x_test.shape)

        # Reshape
        dim1, dim2, dim3 = x_test.shape
        x_test = x_test.reshape(-1, dim3)

        # Call model in loop cause inefficient, must reshape first
        x_test = word2vec.predict(x_test)
        # x_test = [word2vec.predict(x) for x in x_test]

        # Reshape
        x_test = np.array(x_test, dtype=float)
        x_test = x_test.reshape(dim1, dim2, -1)
        # print(x_test.shape)
        return self.model.predict(x_test, verbose=verbose)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path, word2vec):
        self.word2vec = word2vec
        self.model = load_model(path)