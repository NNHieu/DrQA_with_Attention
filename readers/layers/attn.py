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
    def __init__(self, num_head, qk_feature_size, v_feature_size, out_size):
        super(Multihead, self).__init__()
        self.num_head = num_head
        self.qk_feature_size = qk_feature_size
        self.v_feature_size = v_feature_size
        self.proj_Q = LinearWithChannel(qk_feature_size, qk_feature_size, num_head)
        self.proj_K = LinearWithChannel(qk_feature_size, qk_feature_size, num_head)
        self.proj_V = LinearWithChannel(v_feature_size, v_feature_size, num_head)

        self.heads = nn.ModuleList()
        self.linear_out = nn.Linear(num_head*v_feature_size, out_size)
        for i in range(num_head):
            self.heads.append(ScaleDotProductAttention(qk_feature_size, False))
    def forward(self, Q, K, V, K_mask):
        assert K.size(1) == V.size(1)
        num_batch = Q.size(0)
        dq    = Q.size(2)
        dk    = K.size(2)
        dv    = V.size(2)
        qlen  = Q.size(1)
        klen  = vlen = K.size(1)
        assert dq == dk and self.qk_feature_size == dq and self.v_feature_size == dv
        assert len(K_mask.size()) == 2 and K_mask.size(0) == num_batch and K_mask.size(1) == klen


        Q_proj = Q.view(-1, Q.view(-1))
        Q_proj = Q_proj.unsqueeze(0).expand(self.num_head, *Q_proj.size())
        assert Q_proj.size(0) == self.num_head and Q_proj.size(1) == num_batch*qlen and Q_proj.size(2) == dq
        Q_proj = self.proj_Q(Q_proj)
        assert Q_proj.size(0) == self.num_head and Q_proj.size(1) == num_batch*qlen and Q_proj.size(2) == dq


        K_proj = K.view(-1, Q.view(-1))
        K_proj = K_proj.unsqueeze(0).expand(self.num_head, *K_proj.size())
        assert K_proj.size(0) == self.num_head and K_proj.size(1) == num_batch*klen and K_proj.size(2) == dk
        K_proj = self.proj_K(K_proj)
        assert K_proj.size(0) == self.num_head and K_proj.size(1) == num_batch*klen and K_proj.size(2) == dk

        V_proj = V.view(-1, Q.view(-1))
        V_proj = V_proj.unsqueeze(0).expand(self.num_head, *V_proj.size())
        assert V_proj.size(0) == self.num_head and V_proj.size(1) == num_batch*vlen and V_proj.size(2) == dv
        V_proj = self.proj_V(V_proj)
        assert V_proj.size(0) == self.num_head and V_proj.size(1) == num_batch*vlen and V_proj.size(2) == dv


        Q_proj = Q_proj.view(-1, qlen, Q_proj.size(-1))
        assert Q_proj.size(0) == self.num_head*num_batch and Q_proj.size(1) == qlen and Q_proj.size(2) == dq
      
        K_proj = K_proj.view(-1, klen, K_proj.size(-1))
        assert K_proj.size(0) == self.num_head*num_batch and K_proj.size(1) == klen and K_proj.size(2) == dk

        V_proj = V_proj.view(-1, vlen, V_proj.size(-1))
        assert V_proj.size(0) == self.num_head*num_batch and V_proj.size(1) == vlen and V_proj.size(2) == dv

        # Compute scores and scale
        scores = Q_proj.bmm(K_proj.transpose(2, 1)) / math.sqrt(self.qk_feature_size)
        assert scores.size(0) == self.num_head*num_batch and scores.size(1) == qlen and scores.size(2) == klen
        # Mask padding
        K_mask = K_mask.unsqueeze(1).expand(K.size(0), qlen, klen).repeat(self.num_head, 1, 1)
        assert K_mask.size() == scores.size()
        
        scores.data.masked_fill_(K_mask.data.eq(1), -float('inf'))
        # Normalize
        alpha_flat = F.softmax(scores.view(-1, scores.size(-1)), dim=-1)
        alpha = alpha_flat.view(-1, qlen, klen) 
        assert alpha.size(0) == self.num_head*num_batch and alpha.size(1) == qlen and alpha.size(2) == klen

        # Tính attn vô value
        heads = alpha.bmm(V_proj).view(self.num_head, -1, qlen, klen) # shape(num_head, batch, qlen, dv)
        
        head_cat = torch.cat((heads[i, ...] for i in range(self.num_head)), dim=-1)
        out = self.linear_out(head_cat.view(-1, head_cat.size(-1)))
        return out.view(num_batch, qlen, dv)

class SelfAttnMultihead(Multihead):
    def __init__(self, num_head, feature_size, out_size=None):
        super().__init__(num_head, feature_size, feature_size, out_size if out_size is not None else feature_size)

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