import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath( os.path.join(current_dir, "../../") )
sys.path.append(grandparent_dir)

from lib import encoder, embedding, models
from sklearn.model_selection import train_test_split
import numpy as np



# Check for GPU
models.check_gpu()



# Hyperparameters
epochs = 1
batch_size = 128

random_state = 87
validate_size = 0.1

suffix = "_241126"
model_path = os.path.join(current_dir, "pretrained/DeepAFP" + suffix + ".model")



# Load data
pos_data_path = current_dir + "/data/train_pos_data.npy"
neg_data_path = current_dir + "/data/train_neg_data.npy"
pos_data = np.load(pos_data_path)
neg_data = np.load(neg_data_path)

posTAPE_path = current_dir + "/data/train_posTAPE.npy"
negTAPE_path = current_dir + "/data/train_negTAPE.npy"
posTAPE = np.load(posTAPE_path)
negTAPE = np.load(negTAPE_path)

X_data, y_data = encoder.GetLebel(pos_data, neg_data)
y_data = encoder.OneHot2Label(y_data)
X_tape = np.concatenate((posTAPE, negTAPE), axis=0)

del pos_data, neg_data, posTAPE, negTAPE



# Train-Vaidation split
X_tape_train, X_tape_validate, X_data_train, X_data_validate, y_data_train, y_data_validate = train_test_split(X_tape, X_data, y_data, stratify=y_data, test_size=validate_size, random_state=random_state)



# Train model
print("Training model...")
model = models.DeepAFP()
model.fit(X_tape, X_data, y_data, validation_data=([X_tape_validate, X_data_validate], y_data_validate), epochs=epochs, batch_size=batch_size)



# Save model
model.save(model_path)
print("Models saved:", model_path)