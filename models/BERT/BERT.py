# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn

def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)            

    def forward(self, input_ids, speaker_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if speaker_ids is not None:
            speaker_embeddings = self.speaker_embeddings(speaker_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings + speaker_embeddings
        else:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, speaker_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = attention_mask.unsqueeze(1)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, speaker_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class TurnLevelLSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 lstm_dropout,
                 dropout_rate):
        super(TurnLevelLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm2hiddnesize = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        lstm_out = self.lstm(inputs)
        lstm_out = lstm_out[0].squeeze(0)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.bilstm2hiddnesize(lstm_out)
        return lstm_out

class GTN(nn.Module):
    '''adapt from https://github.com/seongjunyun/Graph_Transformer_Networks '''
    def __init__(self, etypes, num_channels, w_in, w_out, num_layers, norm=True, num_class=None, num_attention_heads=16):
        super(GTN, self).__init__()
        self.etypes = etypes
        self.num_edge = len(etypes)+1
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(self.num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(self.num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.attn_encoder = None
        # self.attention_head_size = int(w_in / num_attention_heads)
        # self.attn_encoder = MultiHeadAttention(num_attention_heads, w_in, self.attention_head_size, self.attention_head_size, 0.1)
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
    
        if num_class is not None:
            self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
            self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)).to(H.device)
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)).to(H.device) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)
        deg = torch.sum(H, dim=1).to(H.device)
        # deg_inv = deg.pow(-1)
        # deg_inv[deg_inv == float('inf')] = 0
        # deg_inv = torch.linalg.inv(deg)
        # deg_inv = torch.inverse(deg)
        # deg_inv = (deg + 1e-8) ** -1
        deg_inv = (deg + 1e-8).pow(-1)
        deg_inv[deg==0] = 0
        # y = torch.zeros_like(deg,dtype=deg.dtype)
        # deg_inv = torch.where(deg!=0, (deg+1e-8)**-1, y)
        # if torch.any(torch.isnan(deg_inv)):
        #     print('-------deg--------')
        #     print(deg)
        #     print(deg_inv)
        deg_inv[deg_inv == float('inf')] = 0

        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H
    
    def adj(self, graph):
        num_nodes = graph.number_of_nodes('node')
        for k, etype in enumerate(self.etypes):
            A_k = graph.adj(etype=etype)
            if k==0:
                A = A_k.type(torch.FloatTensor).unsqueeze(-1).to_dense()
            else:
                A = torch.cat([A,A_k.type(torch.FloatTensor).unsqueeze(-1).to_dense()], dim=-1)

        A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

        return A

    def forward(self, graph, X, target_x=None, target=None):
        A = self.adj(graph)
        # print('are adj matrices nan:',torch.any(torch.isnan(A)))
        # print('Adjecency Matrix',A.shape)
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                # print('before norm:',torch.max(abs(H)))
                H = self.normalization(H)
                # print('after norm:',torch.max(abs(H)))

                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        for i in range(self.num_channels):
            if i==0:
                if self.attn_encoder is None:
                    X_ = F.relu(self.gcn_conv(X,H[i]))
                else:
                    X = X.unsqueeze(0)
                    H_ = self.norm(H[i], add=True).unsqueeze(0)
                    X_ = F.relu(self.attn_encoder(X,X,X,H_ ,fill=False)[0].squeeze(0))

                # print('after first gcnconv:',torch.any(torch.isnan(X_)))

            else:
                if self.attn_encoder is None:
                    X_tmp = F.relu(self.gcn_conv(X,H[i]))
                else:
                    H_ = self.norm(H[i], add=True).unsqueeze(0)
                    X_tmp = F.relu(self.attn_encoder(X,X,X,H_,fill=False)[0].squeeze(0))
                X_ = torch.cat((X_,X_tmp), dim=1)
                # print('after {}th gcnconv:'.format(i),torch.any(torch.isnan(X_tmp)))
        if target is None:

            return X_, Ws
        else:
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_[target_x])
            loss = self.loss(y, target)
            return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        # print('is conv1 weight nan:', torch.any(torch.isnan(self.conv1.weight)))
        # print('conv1 gradients',self.conv1.weight.grad)
        # if self.first == True:
            # print('is conv2 weight nan:', torch.any(torch.isnan(self.conv2.weight)))
            # print('conv2 gradients',self.conv2.weight.grad)

        if self.first == True:
            # print('before first gtconv:',torch.any(torch.isnan(A)))
            a = self.conv1(A)
            b = self.conv2(A)
            # print('after first gtconv a:',torch.any(torch.isnan(a)))
            # print('after first gtconv b:',torch.any(torch.isnan(b)))

            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
            # print('after first gtlayer:',torch.any(torch.isnan(H)))
        else:
            # print('before second gtconv:',torch.any(torch.isnan(A)))
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
            # print('after second gtlayer:',torch.any(torch.isnan(H)))

        
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.weight)
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = A.to(self.weight.device)
        
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)

        return A

class TUCOREGCN_BERT(nn.Module):
    def __init__(self, config, num_labels, gtn_layers=2, gtn_out=1024, activation='relu', gtn_num_channels=3, gtn_num_layers=2, gcn_dropout=0.6):
        super(TUCOREGCN_BERT, self).__init__()
        self.bert = BertModel(config)

        self.gcn_dim = config.hidden_size
        self.gcn_layers = gtn_layers
        self.num_labels = num_labels

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size*3*(self.gcn_layers+1), self.num_labels)
        # self.classifier = nn.Linear(config.hidden_size*3*gtn_num_channels, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # self.turnAttention = MultiHeadAttention(config.num_attention_heads, config.hidden_size, self.attention_head_size, self.attention_head_size, config.attention_probs_dropout_prob)

        # rel_name_lists = ['speaker', 'dialog', 'entity']
        # self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
        #                                                    num_bases=len(rel_name_lists), activation=self.activation,
        #                                                    self_loop=True, dropout=gcn_dropout)
        #                                  for i in range(self.gcn_layers)])
        
        # self.LSTM_layers = nn.ModuleList([TurnLevelLSTM(config.hidden_size, 2, 0.2, 0.4) for i in range(self.gcn_layers)])
        
        # # GTN
        # self.node_project = False
        # self.etypes = ['dialog','sequence','entity','speaker']
        # # self.etypes = ['dialog','sequence','subject','object']
        # self.gtn = GTN(self.etypes, gtn_num_channels, config.hidden_size, config.hidden_size, gtn_num_layers)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        speaker_ids=None, 
        graphs=None, 
        mention_id=None,
        turn_mask=None,
        get_node_attention=False, 
        get_semantic_attention=False,
        cls_indices=None,
        tran_ids=None, 
        dataset='DialogRE'
    ):        
        slen = input_ids.size(1)
        bs = cls_indices.shape[0]

        
        # CLS_attn_mask = np.full((bs,slen,slen),0)
        
        # for i in range(bs):
        #     head_cls = 1
        #     len_tokens = (attention_mask[i] == 1).sum()
        #     for j, cls in enumerate(cls_indices[i]):
        #         if j > 1 and cls:
        #             CLS_attn_mask[i, head_cls, head_cls:j] = 1
        #             head_cls = j
        #         elif not cls or j == 0:
        #             CLS_attn_mask[i, j, :len_tokens] = 1

        #     CLS_attn_mask[i, head_cls, head_cls:len_tokens] = 1

        # CLS_attn_mask = torch.from_numpy(CLS_attn_mask).to(input_ids.device)
        CLS_attn_mask = attention_mask
        speaker_ids = None
        sequence_outputs, pooled_outputs = self.bert(input_ids, speaker_ids, token_type_ids, CLS_attn_mask) 
        '''
        graph_output = list()
        fea_idx = 0

        for i,graph in enumerate(graphs):

            # add CLS for each turn
            sequence_output = sequence_outputs[i]
            mask = cls_indices[i].unsqueeze(-1).expand(-1, sequence_output.shape[-1])
            features_graph = torch.masked_select(sequence_output, mask).view(-1, sequence_output.shape[-1])
            # print('features_graph',features_graph.shape)
            node_num = graphs[i].number_of_nodes('node')
            # features_graph = features[fea_idx: fea_idx+node_num]
            if self.node_project:
                dialog = self.dialog_node(features_graph[0].unsqueeze(0))
                turns = self.turn_node(features_graph[1:-2])
                entities = self.entity_node(features_graph[-2:])
                features_graph = torch.cat([dialog,turns,entities])
            features_graph, Ws = self.gtn(graph, features_graph)
            dialog_node = features_graph[0]

            obj_node, sbj_node = features_graph[-2], features_graph[-1]
            intergrated_output = torch.cat((dialog_node,obj_node,sbj_node), dim=-1)
            graph_output.append(intergrated_output)
            # print('after gtn:',i, torch.any(torch.isnan(intergrated_output)))
            fea_idx += node_num
      

        graph_output = torch.stack(graph_output)
        batch_semantic_attn, batch_node_attn = None, None

        pooled_output = self.dropout(graph_output)
        '''
        pooled_output = sequence_outputs[:,0]
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, self.num_labels)
            loss = loss_fct(logits, labels)
            return loss, logits, None, None, None
        else:
            return logits, None, None, None

