import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import Module, init


class PersonalSelfAttentionBase(Module):
    def __init__(self, input_features, expansion_features, epsilon=0,
            affine_output=True, is_in_residual=False, distill_init=True, **kwargs):
        super().__init__()

        self.input_features = input_features
        self.expansion_features = expansion_features
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

        v_weight_shape = (self.input_features, self.expansion_features)
        self.v_weight = Parameter(torch.empty(v_weight_shape))

        q_weight_shape = (self.input_features, self.expansion_features)
        self.q_weight = Parameter(torch.empty(q_weight_shape))
        q_bias_shape = (self.input_features, self.expansion_features)
        self.q_bias = Parameter(torch.empty(q_bias_shape))

        if affine_output:
            self.o_weight = Parameter(torch.empty(self.input_features))
            self.o_bias = Parameter(torch.empty(self.input_features))
            with torch.no_grad():
                self.o_weight.fill_(1)
                self.o_bias.fill_(0.0)
        else:
            self.o_weight = None
            self.o_bias = None

        self.reset_parameters(is_in_residual, distill_init=distill_init)

    def extra_repr(self):
        s = ('input_features={input_features}, '
             'expansion_features={expansion_features}')
        return s.format(**self.__dict__) + f', epsilon={self.epsilon.item()}'

    def reset_parameters(self, is_in_residual, distill_init=True):
        if not is_in_residual:
            # We want a unit variance across it
            # Do a Xavier based init due to softmax acting like a pseudo-sigmoid function
            bound = math.sqrt(6 / (1 + self.expansion_features))
            with torch.no_grad():
                init.uniform_(self.v_weight, -bound, bound)
        else:
            # If doing a distillation with knowledge transfer a much wider distribution converges faster
            if distill_init:
                with torch.no_grad():
                    init.normal_(self.v_weight, 0, 4)
            else:
                bound = math.sqrt(2 / (1 + self.expansion_features))
                with torch.no_grad():
                    init.normal_(self.v_weight, 0, bound)

        with torch.no_grad():
            self.q_weight.fill_(1.0)

            if distill_init:
                init.uniform_(self.q_bias, -1, 1)
            else:
                # init.normal_(self.q_bias, 0, 1)
                init.uniform_(self.q_bias, -3, 3)

        with torch.no_grad():
            if self.o_weight is not None:
                self.o_weight.fill_(1)
            if self.o_bias is not None:
                self.o_bias.fill_(0.0)

    def forward(self, x):
        raise NotImplementedError
