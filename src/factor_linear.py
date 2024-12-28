import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class FactorLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    U: torch.Tensor
    V: torch.Tensor

    def __init__(self, U: nn.Parameter, V: nn.Parameter, bias: nn.Parameter=None, transpose:bool=False) -> None:
        super().__init__()
        
        self.U = U
        self.V = V
        self.transpose = transpose
        if self.transpose:
            self.in_features = self.U.data.shape[0]
            self.out_features = self.V.data.shape[1]
        else:
            self.in_features = self.V.data.shape[1]
            self.out_features = self.U.data.shape[0]

        if bias is not None:
            self.bias = copy.deepcopy(bias)
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            intermediate = F.linear(input, self.U.T)
            output = F.linear(intermediate, self.V.T, self.bias)
        else:
            intermediate = F.linear(input, self.V)
            output = F.linear(intermediate, self.U, self.bias)
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'