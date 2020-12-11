import torch
import torch.nn as nn
import math

class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, input_size, output_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, 1, output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.w, self.b)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        """
        parameter
        ----------
            x: (num_channel, batch_size, feature_size)
        """
        return torch.bmm(x, self.w) + self.b
