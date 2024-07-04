import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def file_to_str(fp):
    with open(fp, "r") as f:
        return f.read()


def import_module_from_string(name: str, source: str):
    """
    Code from https://stackoverflow.com/questions/5362771/how-to-load-a-module-from-code-in-a-string
    Import module from source string.
    Example use:
    import_module_from_string("m", "f = lambda: print('hello')")
    m.f()
    """
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)
    return module


class SentenceEmbedder(nn.Module):
    def __init__(self, device, model_name="prajjwal1/bert-small"):
        super(SentenceEmbedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def encode(self, sentences):
        # code from https://huggingface.co/sentence-transformers/all-mpnet-base-v2

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
