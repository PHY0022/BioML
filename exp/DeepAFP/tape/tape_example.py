import torch
from tape import ProteinBertModel, TAPETokenizer
# pip install tape_proteins


model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

# Pfam Family: Hexapep, Clan: CL0536
sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
token_ids = torch.tensor([tokenizer.encode(sequence)])
print(token_ids)
output = model(token_ids)
sequence_output = output[0]
pooled_output = output[1]

# NOTE: pooled_output is *not* trained for the transformer, do not use
# w/o fine-tuning. A better option for now is to simply take a mean of
# the sequence output

import numpy as np
sequence_output = sequence_output.detach().numpy()
print(sequence_output.shape)
# (1, 38, 768)
print(np.mean(sequence_output, axis=1).shape)  # sequence_output is a tensor with shape (1, L, 768), where L is the length of the sequence
# (1, 768)

pooled_output = pooled_output.detach().numpy()
print(pooled_output.shape)
# (1, 768)

print(sequence_output.all() == np.mean(sequence_output, axis=1).all())
# True