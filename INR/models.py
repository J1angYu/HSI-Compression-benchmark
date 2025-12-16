import torch
from torch import nn

import numpy as np


### SPINR
class SPINR(nn.Module):
    def __init__(self,
                 table_length, 
                 table_dim, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 first_omega=30, 
                 hidden_omega=30.0):

        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((table_length, table_dim)) * 2 - 1))

        self.net = Siren(in_features=table_dim,
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                        out_features=out_features,
                        first_omega=first_omega,
                        hidden_omega=hidden_omega)

    def forward(self, coords):
        output = self.net(self.table)
        return output
    

### SIREN
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                             np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        out = torch.sin(self.omega * self.linear(input))
        return out

class Siren(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 first_omega=30, 
                 hidden_omega=30.0):

        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega=first_omega))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega=hidden_omega))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega,
                                        np.sqrt(6 / hidden_features) / hidden_omega)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


### PEMLP
class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) 
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, input):
        out = [input]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * input)]
        return torch.cat(out, -1)
    
class PEMLP(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 N_freqs=10):
        super().__init__()
        self.enconding = PositionalEncoding(in_channels=in_features, N_freqs=N_freqs)
        
        self.net = []
        self.net.append(nn.Linear(self.enconding.out_channels, hidden_features))
        self.net.append(nn.ReLU(True))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(True))

        final_linear = nn.Linear(hidden_features, out_features)                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(self.enconding(coords))
        return output


### FINER 
class FINERLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30, is_first=False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                            np.sqrt(6 / self.in_features) / self.omega)
    
    def forward(self, input):
        x = self.linear(input)
        with torch.no_grad():
            alpha = torch.abs(x) + 1 
        return torch.sin(self.omega * alpha * x)

class FINER(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 first_omega=30, 
                 hidden_omega=30.0):

        super().__init__()

        self.net = []
        self.net.append(FINERLayer(in_features, hidden_features, 
                                  is_first=True, omega=first_omega))

        for i in range(hidden_layers):
            self.net.append(FINERLayer(hidden_features, hidden_features,
                                      is_first=False, omega=hidden_omega))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega,
                                        np.sqrt(6 / hidden_features) / hidden_omega)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

### Gauss 
class GaussLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_last=False, scale=30.0):
        super().__init__()
        self.scale = scale
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return torch.exp(-(self.scale*wx_b)**2)
        return wx_b
    
class Gauss(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, scale=30):
        super().__init__()
        self.net = []
        self.net.append(GaussLayer(in_features, hidden_features, is_last=False, scale=scale))

        for i in range(hidden_layers):
            self.net.append(GaussLayer(hidden_features, hidden_features, is_last=False, scale=scale))
            
        self.net.append(GaussLayer(hidden_features, out_features, is_last=True, scale=scale))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output