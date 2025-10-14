import torch
import json
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config

SEQ_LEN = 128

class MLMWrapper:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.stoi = {s:i for i,s in enumerate(vocab)}
        self.itos = {i:s for i,s in enumerate(vocab)}

    def encode(self, s, seq_len=SEQ_LEN):
        tokens = []
        i = 0
        while i < len(s):
            if s[i] == "[":
                j = s.find("]", i)
                if j != -1:
                    tok = s[i:j+1]  
                    if tok in self.stoi:
                        tokens.append(tok)
                        i = j+1
                        continue
            tokens.append(s[i])
            i += 1
        ids = [self.stoi.get(tok, self.stoi["[UNK]"]) for tok in tokens]
        if len(ids) < seq_len:
            ids = ids + [self.stoi["[PAD]"]] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]
        return torch.tensor([ids]).long()

    def mask_positions(self, encoded):
        mask_id = self.stoi["[MASK]"]
        return (encoded[0] == mask_id).nonzero(as_tuple=False)

def load_hf_gpt2_model(model_path="./hf_gpt2_model", vocab_path="vocab.json"):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    model.eval()

    return MLMWrapper(model, vocab)

def fill_mask_hf(wrapper, text_with_mask, seq_len=SEQ_LEN):
    device = next(wrapper.model.parameters()).device
    idx = wrapper.encode(text_with_mask, seq_len=seq_len).to(device)
    mask_token_id = wrapper.stoi["[MASK]"]
    mask_pos = (idx[0]==mask_token_id).nonzero(as_tuple=False)

    if mask_pos.numel() == 0:
        raise ValueError("No [MASK] in text")

    with torch.no_grad():
        outputs = wrapper.model(input_ids=idx)
        logits = outputs.logits  

    pos = mask_pos[0,0].item()
    logits_for_pos = logits[0, pos]  
    return logits_for_pos.detach().cpu()

def fill_mask(wrapper, text_with_mask, seq_len=SEQ_LEN):
    return fill_mask_hf(wrapper, text_with_mask, seq_len)