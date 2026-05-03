# models/blocks.py
# Blocos primitivos de convolucao reutilizados pelos encoders - MO434
#
#   conv_block  - convolucao padrao ou depthwise-separavel com BN + ReLU
#   ResBlock    - bloco residual: y = F(x) + x (skip connection)

import torch.nn as nn


def conv_block(c_in, c_out, stride=1, dw=False):
    """
    Bloco de convolucao com batchnorm e relu.

    Se dw=True: usa convolucao depthwise-separavel (muito mais eficiente).
    Convolucao depthwise-separavel: 1 filtro por canal + mistura de canais separada.
    Reduz operacoes em ~8x comparado a conv padrao com mesma capacidade.
    """
    if dw:
        # depthwise separavel: conv por canal + pointwise para mistura de canais
        return nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )


class ResBlock(nn.Module):
    """
    Bloco residual: y = F(x) + x (skip connection).
    Melhora o fluxo de gradiente em redes profundas,
    permitindo treinar redes mais profundas sem degradacao.
    """
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + x)  # adiciona a entrada original (residual)
