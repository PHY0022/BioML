import ankh
import torch
# pip install ankh

# To load large model:
model, tokenizer = ankh.load_large_model()
model.eval()


# To load base model.
model, tokenizer = ankh.load_base_model()
model.eval()


model, tokenizer = ankh.load_large_model()
model.eval()

protein_sequences = [
    'MKALCLLLLPVLGLLVSSKTLCSMEEAINERIQEVAGSLIFRAISSIGLECQSVTSRGDLATCPRGFAVTGCTCGSACGSWDVRAETTCHCQCAGMDWTGARCCRVQPLEHHHHHH', 
    'GSHMSLFDFFKNKGSAATATDRLKLILAKERTLNLPYMEEMRKEIIAVIQKYTKSSDIHFKTLDSNQSVETIEVEIILPR',
    'HLLGRPREALSTNECKARRAASAATAAPTAT',
    ]
print('protein 1 length:', len(protein_sequences[0])) # 116
print('protein 2 length:', len(protein_sequences[1])) # 80

protein_sequences = [list(seq) for seq in protein_sequences]


outputs = tokenizer.batch_encode_plus(protein_sequences, 
                                add_special_tokens=True, 
                                padding=True, 
                                is_split_into_words=True, 
                                return_tensors="pt")
with torch.no_grad():
    embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

print(embeddings.last_hidden_state.detach().numpy().shape)