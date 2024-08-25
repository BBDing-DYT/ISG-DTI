import math

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_len, value=0.0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, num_attention_score_hiddens, dropout):
        super().__init__()
        self.W_q = nn.LazyLinear(num_attention_score_hiddens, bias=False)
        self.W_k = nn.LazyLinear(num_attention_score_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class SimpleCrossMultiHeadAttention(nn.Module):
    def __init__(self, num_heads,
                 num_hiddens_left, num_hiddens_right,
                 num_attention_score_hiddens, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention_left = AdditiveAttention(num_attention_score_hiddens, dropout)
        self.attention_right = AdditiveAttention(num_attention_score_hiddens, dropout)
        self.W_o_left = nn.LazyLinear(num_hiddens_left, bias=bias)
        self.W_o_right = nn.LazyLinear(num_hiddens_right, bias=bias)

    def forward(self, queries_left, keys_right, values_right, valid_lens_right,
                queries_right, keys_left, values_left, valid_lens_left):
        queries_left = self.transpose_qkv(queries_left)
        keys_right = self.transpose_qkv(keys_right)
        values_right = self.transpose_qkv(values_right)
        queries_right = self.transpose_qkv(queries_right)
        keys_left = self.transpose_qkv(keys_left)
        values_left = self.transpose_qkv(values_left)

        if valid_lens_left is not None:
            valid_lens_left = torch.repeat_interleave(valid_lens_left, repeats=self.num_heads, dim=0)
        if valid_lens_right is not None:
            valid_lens_right = torch.repeat_interleave(valid_lens_right, repeats=self.num_heads, dim=0)

        output_left = self.attention_left(queries_left, keys_right, values_right, valid_lens_right)
        output_right = self.attention_right(queries_right, keys_left, values_left, valid_lens_left)
        output_left = self.W_o_left(self.transpose_output(output_left))
        output_right = self.W_o_right(self.transpose_output(output_right))
        return output_left, output_right

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len, dropout):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class WordEmbeddingAndPosEncoding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len, dropout)

    def forward(self, input_ids):
        return self.pos_encoding(self.embedding(input_ids) * math.sqrt(self.hidden_size))


class AddAndLayerNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class FFNLayer(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.activation = nn.GELU(approximate='tanh')
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.activation(self.dense1(X)))


class SimpleCrossTransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads,
                 num_hiddens_left, ffn_num_hiddens_left,
                 num_hiddens_right, ffn_num_hiddens_right,
                 num_attention_score_hiddens, dropout, use_bias=False):
        super().__init__()
        self.attention = SimpleCrossMultiHeadAttention(num_heads,
                                                 num_hiddens_left, num_hiddens_right,
                                                 num_attention_score_hiddens, dropout, use_bias)
        self.addnorm1_left = AddAndLayerNorm(num_hiddens_left, dropout)
        self.addnorm1_right = AddAndLayerNorm(num_hiddens_right, dropout)
        self.ffn_left = FFNLayer(ffn_num_hiddens_left, num_hiddens_left)
        self.ffn_right = FFNLayer(ffn_num_hiddens_right, num_hiddens_right)
        self.addnorm2_left = AddAndLayerNorm(num_hiddens_left, dropout)
        self.addnorm2_right = AddAndLayerNorm(num_hiddens_right, dropout)

    def forward(self, X_left, valid_lens_left, X_right, valid_lens_right):
        Y_last_left = X_left
        Y_last_right = X_right
        Y_left, Y_right = self.attention(Y_last_left, Y_last_right, Y_last_right, valid_lens_right,
                                         Y_last_right, Y_last_left, Y_last_left, valid_lens_left)
        Y_last_left = self.addnorm1_left(Y_last_left, Y_left)
        Y_last_right = self.addnorm1_right(Y_last_right, Y_right)
        return (self.addnorm2_left(Y_last_left, self.ffn_left(Y_last_left)),
                self.addnorm2_right(Y_last_right, self.ffn_right(Y_last_right)))


class NormalGAT(nn.Module):
    def __init__(self, gnn_layer_num, in_channels, num_heads, bias=True):
        super().__init__()
        self.gnn_layers = nn.Sequential()
        for i in range(gnn_layer_num):
            self.gnn_layers.add_module("GNN Layer: " + str(i),
                                       GATConv(in_channels, in_channels // num_heads, num_heads, bias=bias))

    def forward(self, features, edge_indexs):
        for gnn_single_layer in self.gnn_layers:
            features = gnn_single_layer(features, edge_indexs)
        features = F.gelu(features, approximate='tanh')
        return features


class SAPBlock(nn.Module):
    def __init__(self, hidden_size, query_num):
        super().__init__()
        self.W_p1 = nn.LazyLinear(hidden_size, False)
        self.W_p2 = nn.LazyLinear(query_num, False)

    def forward(self, x, valid_lens):
        attention_matrix = self.W_p2(F.tanh(self.W_p1(x)))
        mask = valid_lens.unsqueeze(-1).unsqueeze(-1).expand_as(attention_matrix)
        a_tensor = (torch.arange(1, attention_matrix.size(1) + 1, dtype=torch.float32, device=x.device)
                    .repeat(attention_matrix.size(0), attention_matrix.size(-1), 1).transpose(-1, -2))
        mask = a_tensor > mask
        attention_matrix[mask] = -1e6
        attention_matrix = F.softmax(attention_matrix, dim=-2)
        output = torch.bmm(x.permute(0, 2, 1), attention_matrix)
        return output.permute(0, 2, 1)
