import os
import subprocess
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Bidirectional, LSTM, concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal
import gc
from . import encoder

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    else:
        print("No GPU detected. Use CPU? (y/n)")
        using_cpu = input()
        if using_cpu.lower() != 'y':
            raise RuntimeError("No GPU detected. Exiting...")

def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    print(gc.collect()) 

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

    Input shape: (batch_size, 400)

    Output shape: (batch_size, 400)

    The predict() method is used to get the embeddings (weights of hidden layer) of the input data
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
        model = Sequential([hidden_layer])
        return model.predict(data, verbose=verbose)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        del self.model
        self.model = load_model(path)
        return self

class WE_DL:
    '''
    This model is implemented based on the paper:

    "Incorporating Deep Learning With Word Embedding to Identify Plant Ubiquitylation Sites"
    link: https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2020.572195/full

    A pre-trained Word2Vec model is required

    Input shape: (batch_size, seq_length, 400)
    (For 31mer, seq_length = 31 - 2 + 1 = 30 with kmer size = 2)

    Output shape: (batch_size, 2)
    '''
    def __init__(self, word2vec, optim='adam'):
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
            # Dense(1, activation='sigmoid')
            Dense(2, activation='softmax')
        ])


        if optim.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=0.9)
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def word2vec_predict(self, data, group_size=1000):
        data = np.array(data, dtype=float)
        # print(data.shape)

        # Reshape
        dim1, dim2, dim3 = data.shape
        data = data.reshape(-1, dim3)
        # print(data.shape)

        # data_new = self.word2vec.predict(data)
        for i in range(0, len(data), group_size):
            left = i
            right = i + group_size
            if right > len(data):
                right = len(data)

            # print(f"Predicting {left} to {right}...")
            new = self.word2vec.predict(data[left:right])
            # print(new.shape)
            if i == 0:
                data_new = new
            else:
                data_new = np.concatenate((data_new, new), axis=0)

        del data

        # Reshape
        data_new = np.array(data_new, dtype=float)
        data_new = data_new.reshape(dim1, dim2, -1)

        # print(data_new.shape)
        return data_new

    def fit(self, x_train, y_train, epochs, batch_size, group_size=100, validation_data=None, verbose=1):
        x_train = self.word2vec_predict(x_train, group_size=group_size)
        y_train = np.array(y_train, dtype=float)

        if validation_data is not None:
            x_val, y_val = validation_data
            x_val = self.word2vec_predict(x_val, group_size=group_size)
            y_val = np.array(y_val, dtype=float)
            validation_data = (x_val, y_val)

        self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_data=validation_data)

    def predict(self, x_test, group_size=100, verbose=0):
        x_test = self.word2vec_predict(x_test, group_size=group_size)
        return self.model.predict(x_test, verbose=verbose)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path, word2vec):
        self.word2vec = word2vec
        del self.model
        self.model = load_model(path)
        return self

# class WE_DL_v1:
#     '''
#     Variation of the WE_DL model
#     '''
#     def __init__(self, word2vec, optim='adam'):
#         self.word2vec = word2vec
#         initializer = RandomNormal(mean=0.0, stddev=0.01)
#         self.model = Sequential([
#             Conv1D(128, 3, kernel_initializer=initializer, activation='relu'),
#             MaxPooling1D(2),
#             Dropout(0.5),
            
#             Conv1D(128, 3, kernel_initializer=initializer, activation='relu'),
#             MaxPooling1D(2),
#             Dropout(0.5),
            
#             Conv1D(128, 3, kernel_initializer=initializer, activation='relu'),
#             MaxPooling1D(2),
#             Dropout(0.5),
            
#             Flatten(),
            
#             Dense(128, activation='relu'),
#             Dense(64, activation='relu'),
#             Dense(2, activation='softmax')
#         ])


#         if optim.lower() == 'rmsprop':
#             optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=0.9)
#             self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         else:
#             self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     def word2vec_predict(self, data, group_size=1000):
#         data = np.array(data, dtype=float)
#         # print(data.shape)

#         # Reshape
#         dim1, dim2, dim3 = data.shape
#         data = data.reshape(-1, dim3)
#         # print(data.shape)

#         # data_new = self.word2vec.predict(data)
#         for i in range(0, len(data), group_size):
#             left = i
#             right = i + group_size
#             if right > len(data):
#                 right = len(data)

#             # print(f"Predicting {left} to {right}...")
#             new = self.word2vec.predict(data[left:right])
#             # print(new.shape)
#             if i == 0:
#                 data_new = new
#             else:
#                 data_new = np.concatenate((data_new, new), axis=0)

#         del data

#         # Reshape
#         data_new = np.array(data_new, dtype=float)
#         data_new = data_new.reshape(dim1, dim2, -1)

#         # print(data_new.shape)
#         return data_new

#     def fit(self, x_train, y_train, epochs, batch_size, validation_data=None, verbose=1):
#         x_train = self.word2vec_predict(x_train)
#         y_train = np.array(y_train, dtype=float)

#         if validation_data is not None:
#             x_val, y_val = validation_data
#             x_val = self.word2vec_predict(x_val)
#             y_val = np.array(y_val, dtype=float)
#             validation_data = (x_val, y_val)

#         self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_data=validation_data)

#     def predict(self, x_test, verbose=0):
#         x_test = self.word2vec_predict(x_test)
#         return self.model.predict(x_test, verbose=verbose)
    
#     def save(self, path):
#         self.model.save(path)

#     def load(self, path, word2vec):
#         self.word2vec = word2vec
#         del self.model
#         self.model = load_model(path)
#         return self

class DeepAFP:
    '''
    This model is implemented based on the paper:

    "DeepAFP: An effective computational framework for identifying antifungal peptides based on deep learning"
    link: https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4758?msockid=08ab38e43da369ce130e2cc13c3768bb

    This model has source code on Github:
    link: https://github.com/lantianyao/DeepAFP/tree/main
    '''
    def __init__(self, l=31, cnn_dim=32, kernel_size=3, lstm_dim=30, feature_shape=45, embedding_len=768):
        bert_input = Input(shape=(embedding_len))
        bert_layer = Dense(300, activation='relu')(bert_input)


        feature_input = Input(shape=(l, feature_shape))


        CNN_BiLSTM1 = Conv1D(cnn_dim, kernel_size, padding='same', strides=1, activation='relu')(feature_input)
        CNN_BiLSTM1 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM1)
        CNN_BiLSTM1 = Dropout(0.5)(CNN_BiLSTM1)
        CNN_BiLSTM1 = Bidirectional(LSTM(lstm_dim, return_sequences=True))(CNN_BiLSTM1)
        CNN_BiLSTM1 = Flatten()(CNN_BiLSTM1)
        CNN_BiLSTM1 = Dense(100, activation='relu')(CNN_BiLSTM1)


        CNN_BiLSTM2 = Conv1D(cnn_dim, kernel_size + 1, padding='same', strides=1, activation='relu')(feature_input)
        CNN_BiLSTM2 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM2)
        CNN_BiLSTM2 = Dropout(0.5)(CNN_BiLSTM2)
        CNN_BiLSTM2 = Bidirectional(LSTM(lstm_dim, return_sequences=True))(CNN_BiLSTM2)
        CNN_BiLSTM2 = Flatten()(CNN_BiLSTM2)
        CNN_BiLSTM2 = Dense(100, activation='relu')(CNN_BiLSTM2)


        CNN_BiLSTM3 = Conv1D(cnn_dim, kernel_size + 2, padding='same', strides=1, activation='relu')(feature_input)
        CNN_BiLSTM3 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM3)
        CNN_BiLSTM3 = Dropout(0.5)(CNN_BiLSTM3)
        CNN_BiLSTM3 = Bidirectional(LSTM(lstm_dim, return_sequences=True))(CNN_BiLSTM3)
        CNN_BiLSTM3 = Flatten()(CNN_BiLSTM3)
        CNN_BiLSTM3 = Dense(100, activation='relu')(CNN_BiLSTM3)


        # Feature Fusion Module
        FFM = concatenate([bert_layer, CNN_BiLSTM1, CNN_BiLSTM2, CNN_BiLSTM3], axis=1)
        FFM = Dense(64, activation='relu')(FFM)
        FFM = Dropout(0.5)(FFM)
        FFM = Dense(2, activation='softmax')(FFM)


        self.model = tf.keras.Model(inputs=[bert_input, feature_input], outputs=FFM)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, bert_input, feature_input, y_train, epochs, batch_size, validation_data=None, verbose=1):
        self.model.fit([bert_input, feature_input], y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=verbose)

    def predict(self, bert_input, feature_input, verbose=0):
        return self.model.predict([bert_input, feature_input], verbose=verbose)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        del self.model
        self.model = load_model(path)
        return self

class DeepAFP_v1:
    '''
    Variation of the DeepAFP model
    '''
    def __init__(self, l=31, cnn_dim=32, kernel_size=3, lstm_dim=30, feature_shape=45):
        bert_input = Input(shape=(768))
        bert_layer = Dense(300, activation='relu')(bert_input)


        feature_input = Input(shape=(l, feature_shape))


        CNN_BiLSTM1 = Conv1D(cnn_dim, kernel_size, padding='same', strides=1, activation='relu')(feature_input)
        # CNN_BiLSTM1 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM1) ##
        # CNN_BiLSTM1 = Dropout(0.5)(CNN_BiLSTM1)
        # CNN_BiLSTM1 = Conv1D(cnn_dim, kernel_size, padding='same', strides=1, activation='relu')(CNN_BiLSTM1) ##
        CNN_BiLSTM1 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM1)
        CNN_BiLSTM1 = Dropout(0.5)(CNN_BiLSTM1)
        CNN_BiLSTM1 = Bidirectional(LSTM(lstm_dim, return_sequences=True))(CNN_BiLSTM1)
        CNN_BiLSTM1 = Flatten()(CNN_BiLSTM1)
        CNN_BiLSTM1 = Dense(100, activation='relu')(CNN_BiLSTM1)


        CNN_BiLSTM2 = Conv1D(cnn_dim, kernel_size + 1, padding='same', strides=1, activation='relu')(feature_input)
        # CNN_BiLSTM2 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM2) ##
        # CNN_BiLSTM2 = Dropout(0.5)(CNN_BiLSTM2)
        # CNN_BiLSTM2 = Conv1D(cnn_dim, kernel_size + 1, padding='same', strides=1, activation='relu')(CNN_BiLSTM2) ##
        CNN_BiLSTM2 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM2)
        CNN_BiLSTM2 = Dropout(0.5)(CNN_BiLSTM2)
        CNN_BiLSTM2 = Bidirectional(LSTM(lstm_dim, return_sequences=True))(CNN_BiLSTM2)
        CNN_BiLSTM2 = Flatten()(CNN_BiLSTM2)
        CNN_BiLSTM2 = Dense(100, activation='relu')(CNN_BiLSTM2)


        CNN_BiLSTM3 = Conv1D(cnn_dim, kernel_size + 2, padding='same', strides=1, activation='relu')(feature_input)
        # CNN_BiLSTM3 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM3) ##
        # CNN_BiLSTM3 = Dropout(0.5)(CNN_BiLSTM3)
        # CNN_BiLSTM3 = Conv1D(cnn_dim, kernel_size + 2, padding='same', strides=1, activation='relu')(CNN_BiLSTM3) ##
        CNN_BiLSTM3 = MaxPooling1D(pool_size=2, strides=2)(CNN_BiLSTM3)
        CNN_BiLSTM3 = Dropout(0.5)(CNN_BiLSTM3)
        CNN_BiLSTM3 = Bidirectional(LSTM(lstm_dim, return_sequences=True))(CNN_BiLSTM3)
        CNN_BiLSTM3 = Flatten()(CNN_BiLSTM3)
        CNN_BiLSTM3 = Dense(100, activation='relu')(CNN_BiLSTM3)


        # Feature Fusion Module
        FFM = concatenate([bert_layer, CNN_BiLSTM1, CNN_BiLSTM2, CNN_BiLSTM3], axis=1)
        FFM = Dense(64, activation='relu')(FFM)
        FFM = Dropout(0.5)(FFM)
        FFM = Dense(32, activation='relu')(FFM)
        FFM = Dropout(0.5)(FFM)
        FFM = Dense(2, activation='softmax')(FFM)


        self.model = tf.keras.Model(inputs=[bert_input, feature_input], outputs=FFM)
        # scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=50, decay_rate=0.5, staircase=True)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
        # self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, bert_input, feature_input, y_train, epochs, batch_size, validation_data=None, verbose=1):
        self.model.fit([bert_input, feature_input], y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=verbose)

    def predict(self, bert_input, feature_input, verbose=0):
        return self.model.predict([bert_input, feature_input], verbose=verbose)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        del self.model
        self.model = load_model(path)
        return self