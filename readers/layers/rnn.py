from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

class BRNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size,
                 dropout_rate=0, rnn_unit_type=nn.LSTM,
                 concat_layers=False,):
        super(BRNNBlock, self).__init__()
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
                                      bidirectional=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
    
    def forward(self, x):
        if isinstance(x, PackedSequence):
            h = self.rnn(x)[0]
        else:
            h = self.rnn(x)[0]
        if self.concat_layers:
            return self.dropout(h.data), h
        return self.dropout(h.data)

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
            self.rnns.append(BRNNBlock(input_size, hidden_size, dropout_rate, rnn_unit_type, concat_layers))
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