'''
###########
# DeepAFP #
###########

Implemented based on the paper:

"DeepAFP: An effective computational framework for identifying antifungal peptides based on deep learning"
link: https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4758?msockid=08ab38e43da369ce130e2cc13c3768bb
'''
import sys
import os

# To import lib from grandparent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath( os.path.join(current_dir, "../../") )
sys.path.append(grandparent_dir)

from lib import encoder, models, embedding, evaluate
from sklearn.model_selection import train_test_split
import numpy as np



# Check for GPU
models.check_gpu()



# Hyperparameters
epochs = 100
batch_size = 128

random_state = 87
test_size = 0.2
validate_size = 0.1

pretrained = False
save_model = True
suffix = "_241128_notBalanced"
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



# Train-test-validate split
Idxs = range(len(X_data))
train_idxs, test_idxs, _, _ = train_test_split(Idxs, y_data, shuffle=True, stratify=y_data, test_size=test_size + validate_size, random_state=random_state)
test_idxs, validate_idxs, _, _ = train_test_split(test_idxs, y_data[test_idxs], shuffle=True, stratify=y_data[test_idxs], test_size=validate_size / test_size, random_state=random_state)

X_data_train, y_data_train, = X_data[train_idxs], y_data[train_idxs]
X_data_test, y_data_test = X_data[test_idxs], y_data[test_idxs]
X_data_validate, y_data_validate = X_data[validate_idxs], y_data[validate_idxs]

X_tape_train = X_tape[train_idxs]
X_tape_test = X_tape[test_idxs]
X_tape_validate = X_tape[validate_idxs]



# Train model
if pretrained:
    print("Loading pretrained model...")
    model = models.DeepAFP().load(model_path)
else:
    print("Training model...")
    model = models.DeepAFP()
    model.fit(X_tape_train, X_data_train, y_data_train, validation_data=([X_tape_validate, X_data_validate], y_data_validate), epochs=epochs, batch_size=batch_size)
    if save_model:
        model.save(model_path)



# Evaluate
y_pred = model.predict(X_tape_test, X_data_test)

y_proba = y_pred[:, 0]
y_pred = np.argmax(y_pred, axis=1)

evaluate.Evaluation(y_data_test[:, 0], y_proba)

count_pos = 0
count_neg = 0

for data in y_data_test[:, 1]:
    if data == 1:
        count_pos += 1
    else:
        count_neg += 1

print(f"Positive: {count_pos}, Negative: {count_neg}, Ratio: {count_pos / count_neg}")
print(y_data_test[:5, 1])