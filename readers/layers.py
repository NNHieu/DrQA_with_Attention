from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from .mish import Mish
import math

class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size,
                 dropout_rate=0, rnn_unit_type=nn.LSTM,
                 concat_layers=False, bidirectional=False):
        super(RNNBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.concat_layers = concat_layers
        
        # self.rnns = rnn_unit_type(input_size, 
        #                     hidden_size, 
        #                     num_layers=num_layers, 
        #                     bidirectional=True,
        #                     dropout=dropout_rate)
        
        #https://discuss.pytorch.org/t/how-to-retrieve-hidden-states-for-all-time-steps-in-lstm-or-bilstm/1087/14
        
        self.rnn = rnn_unit_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=self.dropout_rate)
    
    def forward(self, x):
        if isinstance(x, PackedSequence):
            h = self.rnn(x)[0]
        else:
            h = self.rnn(x)[0]
        if self.concat_layers:
            return self.dropout(h.data), h
        return self.dropout(h.data)

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False,
                 dropout_rate=0, dropout_output=False, rnn_unit_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(CustomRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        
        # self.rnns = rnn_unit_type(input_size, 
        #                     hidden_size, 
        #                     num_layers=num_layers, 
        #                     bidirectional=True,
        #                     dropout=dropout_rate)
        
        #https://discuss.pytorch.org/t/how-to-retrieve-hidden-states-for-all-time-steps-in-lstm-or-bilstm/1087/14
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(RNNBlock(input_size, hidden_size, dropout_rate, rnn_unit_type, concat_layers, bidirectional))
        self.dropout_output = self.dropout_output and self.dropout_rate
        if self.dropout_output :
            self.dropout = nn.Dropout(p=self.dropout_rate, inplace=False)
    
    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0 or x_mask.shape[0] == 1:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)
        if self.dropout_output:
            output = self.dropout(output)
        return output.contiguous()

    def _forward_padded(self, x, x_mask):
        """Parameters
            ----------
            x : batch * seq len * input_size
                embedding của input seq
            x_mask: batch * seq len
                    0 or 1 (1 ứng với padding)
            Return
            ----------
            out: batch * seq len * hidden size"""
        
        # Chuẩn bị để pack sequence https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # Độ dài không pad của các seq
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        if ((lengths <= 0).sum() > 0):
            raise Exception
        # Sort seq theo chiều giảm dần độ dài
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        # index dùng để unsort sau 
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        # Sort x
        x = x.index_select(0, idx_sort)
        # Transpose batch and sequence dims vì input của rnn có shape (seq_len, batch, input_size)
        x = x.transpose(0, 1)
        # Pack để làm input cho rnn 
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        
        
        outputs = [rnn_input]
        for i in range(self.num_layers):
            o, h = self.rnns[i](rnn_input)
            rnn_input = nn.utils.rnn.PackedSequence(o,
                                                rnn_input.batch_sizes)
            # Lưu lại hidden states của layer i
            outputs.append(h)
        
        # Unpack
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers hay chỉ lấy layer cuối
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose và unsort ngược về
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad thêm vào seq để giống seq len ban đầu
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, padding], 1)
        return output

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        rnn_input = x
        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input, h = self.rnns[i](rnn_input)
            # Lưu lại hidden states của layer i
            outputs.append(h)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        # Transpose back
        output = output.transpose(0, 1)
        return output

class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * (hdim x num_layer)
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1)) #shape(batch*len, hdim)
        scores = self.linear(x_flat).view(x.size(0), x.size(1)) #shape(batch, len)
        # scores.data.masked_fill_(torch.eq(x_mask, 1), -float('inf'))
        scores.data.masked_fill_(x_mask.data.eq(1), -float('inf'))
        # print(scores.type())
        # print(x_mask)
        # print(scores)
        alpha = F.softmax(scores, dim=-1)
        # print(alpha)
        # raise Exception
        return alpha

class AttnMatch(nn.Module):
    """ Tính trọng số của sequence Y với mỗi phần tử trong X.
    
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(AttnMatch, self).__init__()
        self.input_size = input_size
        if not identity:
            self.linear_x = nn.Linear(input_size, input_size)
            self.linear_y = nn.Linear(input_size, input_size)
            self.linear_v = nn.Linear(input_size, input_size)
        else:
            self.linear_x = None

    def forward(self, Q, K, V, K_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear_x:
            Q_proj = self.linear_x(Q.view(-1, Q.size(-1))).view(Q.size()) 
            K_proj = self.linear_y(K.view(-1, K.size(-1))).view(K.size())
            V_proj = self.linear_v(V.view(-1, K.size(-1))).view(V.size())
        else:
            Q_proj = Q
            K_proj = K
            V_proj = V

        # Compute scores
        scores = Q_proj.bmm(K_proj.transpose(2, 1)) / math.sqrt(self.input_size) # Xp*Yp.T, shape(batch, len1, len2)

        # Mask padding
        K_mask = K_mask.unsqueeze(1).expand(scores.size()) # shape(batch, len1, len2)
        # scores.data.masked_fill_(torch.eq(y_mask, 1), -float('inf'))
        scores.data.masked_fill_(K_mask.data.eq(1), -float('inf'))

        # Normalize bằng softmax, tổng theo dim 2 bằng 1
        alpha_flat = F.softmax(scores.view(-1, K.size(1)), dim=-1) #shape(batch*len1, len2)
        alpha = alpha_flat.view(-1, Q.size(1), K.size(1)) # shape(batch, len1, len2)

        # Take weighted average
        matched_seq = alpha.bmm(V_proj) # shape(batch, len1, hdim)
        
        return matched_seq

class Multihead(nn.Module):
    def __init__(self, hidden_size, out_size, num_head):
        super(Multihead, self).__init__()
        self.num_head = num_head
        self.heads = nn.ModuleList()
        self.linear_out = nn.Linear(num_head*hidden_size, out_size)
        for i in range(num_head):
            self.heads.append(AttnMatch(hidden_size, False))
    def forward(self, Q, K, V, K_mask):
        head_outs = []
        for i in range(self.num_head):
            head_outs.append(self.heads[i](Q, K, V, K_mask))
        head_cat = torch.cat(head_outs, dim=-1)
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

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize
        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None 

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 ứng với padding, 0 là true token)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2) # shape(batch, len)
        # xWy.data.masked_fill_(torch.eq(x_mask, 1), -float('inf'))
        xWy.data.masked_fill_(x_mask.data.eq(1), -float('inf'))
        if self.normalize:
            # if self.training:
            #     print('training')
            #     # In training we output log-softmax for NLL
            #     alpha = F.log_softmax(xWy, dim=-1)
            # else:
                # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha

class MaskLayer(nn.Module):
    def __init__(self) -> None:
        super(MaskLayer, self).__init__()

    def forward(self, x, x_mask, value=-float('inf'), inplace=False):
        if inplace:
            return x.data.masked_fill_(x_mask.data.eq(1), value)
        return x.masked_fill_(x_mask.data.eq(1), value)


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = torch.ones(x.size(0), x.size(1))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)