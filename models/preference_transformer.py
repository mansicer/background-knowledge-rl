import torch
import torch.nn as nn
import torch.nn.functional as F


class PreferenceAttention(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super(PreferenceAttention, self).__init__()
        self.n_embed = n_embed
        self.query_embed = nn.Linear(n_embed, n_embed)
        self.key_embed = nn.Linear(n_embed, n_embed)
        self.value_embed = nn.Linear(n_embed, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, n_embed)
        query = self.query_embed(x)
        key = self.key_embed(x)
        value = self.value_embed(x)
        attn_weights = torch.bmm(query, key.transpose(1, 2)) / (value.shape[-1] ** (1 / 2))
        causal_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        logits = torch.bmm(attn_weights, value)
        logits = torch.sum(logits, dim=1)
        aux_output = {
            "reward_attn_weights": attn_weights,
            "values": value,
        }
        return logits, aux_output


class PreferenceTransformer(nn.Module):
    def __init__(self, state_processor, action_processor, max_seq_len=1000, n_layers: int = 3, n_embed: int = 1, n_headers: int = 128, dropout: float = 0.0):
        super(PreferenceTransformer, self).__init__()
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.timestep_embedding = nn.Embedding(max_seq_len, n_embed)

        self.causal_transformer_layer = nn.TransformerDecoderLayer(d_model=n_embed, nhead=n_headers, dropout=dropout, batch_first=True)
        self.causal_transformer = nn.TransformerDecoder(self.causal_transformer_layer, num_layers=n_layers)
        self.preference_attention = PreferenceAttention(n_embed=n_embed, dropout=dropout)

    def forward_single(self, states, actions, timesteps):
        states, timesteps = states[:, :-1], timesteps[:, :-1]
        # states, actions, timesteps = states[:, :-1], actions[:, :-1], timesteps[:, :-1]
        batch_size, seq_length = states.shape[0], states.shape[1]
        state_embedding = self.state_processor(states)
        action_embedding = self.action_processor(actions)
        embedding = torch.stack([state_embedding, action_embedding], dim=-2)
        timestep_embedding = self.timestep_embedding(timesteps).unsqueeze(-2).repeat(1, 1, 2, 1)
        embedding = embedding + timestep_embedding
        embedding = embedding.reshape(batch_size, seq_length * 2, -1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length * 2, device=embedding.device)
        transformer_output = self.causal_transformer(embedding, embedding, tgt_mask=causal_mask, memory_mask=causal_mask)
        action_output_heads = transformer_output[:, 1::2]
        logits, aux_output = self.preference_attention(action_output_heads)
        aux_output["transformer_hidden"] = transformer_output
        return logits, aux_output

    def forward(self, s1, a1, t1, s2, a2, t2):
        logits1, aux_output1 = self.forward_single(s1, a1, t1)
        logits2, aux_output2 = self.forward_single(s2, a2, t2)
        logits = torch.cat([logits1, logits2], dim=-1)
        values = torch.stack([aux_output1["values"], aux_output2["values"]], dim=-1)
        return logits, values

    def compute_reward(self, states, actions, timesteps):
        logits, aux_output = self.forward_single(states, actions, timesteps)
        return aux_output["values"][:, -1].squeeze(-1), aux_output["transformer_hidden"][:, -1]
