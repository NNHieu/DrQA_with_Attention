from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from .mish import Mish
from .general import LinearWithChannel
import math

class ScaleDotProductAttention(nn.Module):
    """ 
    """
    def __init__(self, input_size, identity=False):
        super(ScaleDotProductAttention, self).__init__()
        self.input_size = input_size
        self.identity = identity
        if not identity:
            self.linear_Q = nn.Linear(input_size, input_size)
            self.linear_K = nn.Linear(input_size, input_size)
            self.linear_V = nn.Linear(input_size, input_size)

    def forward(self, Q, K, V, K_mask):
        """
        Args:
            Q: batch * len1 * hdim
            K: batch * len2 * hdim
            V: batch * len2 * hv_dim
            K_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if not self.identity:
            Q = self.linear_Q(Q.view(-1, Q.size(-1))).view(Q.size()) 
            K = self.linear_K(K.view(-1, K.size(-1))).view(K.size())
            V = self.linear_V(V.view(-1, K.size(-1))).view(V.size())
        # Compute scores and scale
        scores = Q.bmm(K.transpose(2, 1)) / math.sqrt(self.input_size)
        # Mask padding
        K_mask = K_mask.unsqueeze(1).expand(scores.size()) # shape(batch, len1, len2)
        scores.data.masked_fill_(K_mask.data.eq(1), -float('inf'))
        # Normalize
        alpha_flat = F.softmax(scores.view(-1, K.size(1)), dim=-1) #shape(batch*len1, len2)
        alpha = alpha_flat.view(-1, Q.size(1), K.size(1)) # shape(batch, len1, len2)
        # Tính attn vô value
        matched_seq = alpha.bmm(V) # shape(batch, len1, hdim)
        return matched_seq

class Multihead(nn.Module):
    def __init__(self, feature_size, out_size, num_head):
        super(Multihead, self).__init__()
        self.num_head = num_head
        self.feature_size = feature_size
        self.proj_Q = LinearWithChannel(feature_size, feature_size, num_head)
        self.proj_K = LinearWithChannel(feature_size, feature_size, num_head)
        self.proj_V = LinearWithChannel(feature_size, feature_size, num_head)

        self.heads = nn.ModuleList()
        self.linear_out = nn.Linear(num_head*feature_size, out_size)
        for i in range(num_head):
            self.heads.append(ScaleDotProductAttention(feature_size, False))
    def forward(self, Q, K, V, K_mask):
        assert K.size(1) == V.size(1)
        qlen = Q.size(1)
        klen  = vlen = K.size(1)
        
        Q_proj = self.proj_Q(Q.view(-1, Q.view(-1)).unsqueeze(0).expand(self.num_head, *Q.size()))
        K_proj = self.proj_K(K.view(-1, Q.view(-1)).unsqueeze(0).expand(self.num_head, *K.size()))
        V_proj = self.proj_V(V.view(-1, Q.view(-1)).unsqueeze(0).expand(self.num_head, *V.size()))

        Q_proj = Q_proj.view(-1, qlen, Q_proj.size(-1))
        K_proj = K_proj.view(-1, klen, K_proj.size(-1))
        V_proj = V_proj.view(-1, vlen, V_proj.size(-1))
        # Compute scores and scale
        scores = Q_proj.bmm(K_proj.transpose(2, 1)) / math.sqrt(self.feature_size)
        # Mask padding
        K_mask = K_mask.unsqueeze(1).expand(K.size(0), qlen, klen).repeat(self.num_head, 1, 1)
        scores.data.masked_fill_(K_mask.data.eq(1), -float('inf'))
        # Normalize
        alpha_flat = F.softmax(scores.view(-1, K.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, qlen, klen)
        # Tính attn vô value
        heads = alpha.bmm(V).view(self.num_head, -1, qlen, klen) # shape(num_head, batch, len1, hdim)
        
        head_cat = torch.cat(heads[:, ...], dim=-1)
        out = self.linear_out(head_cat.view(-1, head_cat.size(-1)))
        return out.view(head_cat.size(0), head_cat.size(1), int(head_cat.size(2)/self.num_head))

class EncodeModule(nn.Module):
    def __init__(self, hidden_size, out_size, num_head, dropout_rate=0.1) -> None:
        super(EncodeModule, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm1 = nn.LayerNorm((hidden_size, ))
        self.norm2 = nn.LayerNorm((hidden_size, ))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mul_head = Multihead(hidden_size, out_size, num_head)

    def forward(self, Q, K, V, K_mask):
        heads = self.mul_head(Q, K, V, K_mask)
        heads = self.norm1(Q + heads)
        self.dropout(heads)
        heads2 = self.linear1(heads.view(-1, heads.size(-1)))
        heads2 = F.gelu(heads2)
        heads2 = self.linear2(heads2).view(heads.size())
        heads2 = F.gelu(heads2)
        heads2 = self.norm2(heads2 + heads)
        return self.dropout(heads2)