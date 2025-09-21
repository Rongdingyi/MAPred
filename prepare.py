import torch
from esm import Alphabet, pretrained
from esm.data import FastaBatchedDataset
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, T5EncoderModel
import re
import torch.nn as nn
from utils.utils import mutate_single_seq_ECs, retrive_esm1b_embedding, compute_esm_distance

train_file = "split100"
train_fasta_file = mutate_single_seq_ECs(train_file)
esm_model, alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
tokenizer = T5Tokenizer.from_pretrained('./data/prostt5', do_lower_case=False)
di_model = T5EncoderModel.from_pretrained("./data/prostt5")

class ALL_model(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, esm_model, tokenizer, di_model):
        super().__init__()
        self.esm_model = esm_model
        self.tokenizer = tokenizer
        self.di_model = di_model
        self.repr_layers = [(i + self.esm_model.num_layers + 1) % (self.esm_model.num_layers + 1) for i in [-1]]
        self.dtype = torch.float32

    def forward(self, strs, toks):
        truncate_len = min(1022, len(strs[0]))
        out = self.esm_model(toks, repr_layers=self.repr_layers, return_contacts="contacts")
        representations = {layer: t[0, 1 : truncate_len + 1].to(device="cpu") for layer, t in out["representations"].items()}[33]
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in strs]
        sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in sequence_examples]
        ids = self.tokenizer.batch_encode_plus(sequence_examples,
                                        add_special_tokens=True,
                                        padding="longest",
                                        return_tensors='pt').to(toks.device)
        embedding_rpr = self.di_model(ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state[0,1:-1]
        return out, representations, embedding_rpr
    
import os
device = 'cuda:6'
model = ALL_model(esm_model, tokenizer, di_model).to(device)
model.eval()

dataset = FastaBatchedDataset.from_file(train_fasta_file)
batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
)

with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        for i in range(len(labels)):
            if not os.path.exists('./data/esm/' + labels[i] + '.pt') or not os.path.exists('./data/3di/' + labels[i] + '.pt'):
                print('Processing: ', labels[i])
                out, representations, embedding_rpr = model([strs[i]], toks[i].unsqueeze(0).to(device=device, non_blocking=True))
                # print(embedding_rpr)
                torch.save(representations, './data/esm/' + labels[i] + '.pt')
                torch.save(embedding_rpr, './data/3di/' + labels[i] + '.pt')
compute_esm_distance(train_file)