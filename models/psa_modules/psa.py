from .psa_base import PersonalSelfAttentionBase


def psa_func(x, v_weight, q_weight, q_bias, o_weight, o_bias, epsilon):
    # Calculate scores from distance
    q = ((x.unsqueeze(-1) - q_bias)**2) / (
        (q_weight**2) + epsilon)

    # Find attention from scores
    attn = (-q).softmax(dim=-1)

    # Create our basis functions
    v = x.unsqueeze(-1) * v_weight

    # Multiply basis functions with attention and combine
    nl_o = (v * attn).sum(dim=-1)
    nl_o_pre_aff = nl_o.clone()

    # Output affine
    if o_weight is not None:
        nl_o = nl_o * o_weight.view(1, -1)
    if o_bias is not None:
        nl_o = nl_o + o_bias.view(1, -1)

    return attn, v, nl_o_pre_aff, nl_o


class PersonalSelfAttention(PersonalSelfAttentionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        _, _, _, nl_o = psa_func(
            x, self.v_weight, self.q_weight, self.q_bias,
            self.o_weight, self.o_bias, self.epsilon)

        return nl_o
