import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_unit_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(CustomBRNN, self).__init__()
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
            self.rnns.append(rnn_unit_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
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
            rnn_input = outputs[-1]

            # dropout
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            # Lưu lại hidden states của layer i
            outputs.append(self.rnns[i](rnn_input)[0])
        
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

        # Dropout output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
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
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1)) #shape(batch*len, hdim)
        scores = self.linear(x_flat).view(x.size(0), x.size(1)) #shape(batch, len)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha

class SeqAttnMatch(nn.Module):
    """ Tính trọng số của sequence Y với mỗi phần tử trong X.
    
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            # Project hidden dim của x
            x_proj = self.linear(x.view(-1, x.size(-1))).view(x.size()) 
            x_proj = F.relu(x_proj)
            # Project hidden dim của y
            y_proj = self.linear(y.view(-1, y.size(-1))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1)) # Xp*Yp.T, shape(batch, len1, len2)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size()) # shape(batch, len1, len2)
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize bằng softmax, tổng theo dim 2 bằng 1
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1) #shape(batch*len1, len2)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1)) # shape(batch, len1, len2)

        # Take weighted average
        matched_seq = alpha.bmm(y) # shape(batch, len1, hdim)
        return matched_seq


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
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha

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