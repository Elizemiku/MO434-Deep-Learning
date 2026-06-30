# models/encoders.py
# Encoders leves para o student - MO434 (Q3)
#
# Tres variantes para responder Q3 (qual arquitetura de student e melhor):
#   PlainCNNEncoder      - baseline: 4 conv blocks empilhados
#   DepthwiseCNNEncoder  - inspirado no MobileNet, ~8x menos params
#   MiniResNetEncoder    - conv blocks + skip connections para melhor gradiente

import torch.nn as nn
from models.blocks import conv_block, ResBlock


class PlainCNNEncoder(nn.Module):
    """
    Encoder leve: 4 conv blocks empilhados com stride 2 em cada.
    Baseline para comparacao com Depthwise e MiniResNet (Q3).
    Saida: [B, C, H/16, W/16]
    """
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        layers = [conv_block(3, channels[0], stride=2)]
        for i in range(1, len(channels)):
            layers.append(conv_block(channels[i-1], channels[i], stride=2))
        self.features = nn.Sequential(*layers)
        self.out_dim  = channels[-1]

    def forward(self, x):
        return self.features(x)  # saida: [B, C, H/16, W/16]


class DepthwiseCNNEncoder(nn.Module):
    """
    Encoder com convolucos depthwise-separaveis: muito mais leve (~8x menos params).
    Arquitetura inspirada no MobileNet, ideal para dispositivos com restricao de recursos.
    """
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        # primeira camada: conv padrao (input rgb, depthwise nao faz sentido aqui)
        layers = [conv_block(3, channels[0], stride=2, dw=False)]
        for i in range(1, len(channels)):
            layers.append(conv_block(channels[i-1], channels[i], stride=2, dw=True))
        self.features = nn.Sequential(*layers)
        self.out_dim  = channels[-1]

    def forward(self, x):
        return self.features(x)


class MiniResNetEncoder(nn.Module):
    """
    Encoder leve com skip connections.
    Combina a eficiencia dos blocos simples com melhor fluxo de gradiente.
    Bloco residual apos cada downsampling preserva informacao de gradiente.
    """
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        layers = [conv_block(3, channels[0], stride=2)]
        for i in range(1, len(channels)):
            layers.append(conv_block(channels[i-1], channels[i], stride=2))
            layers.append(ResBlock(channels[i]))  # bloco residual apos cada downsampling
        self.features = nn.Sequential(*layers)
        self.out_dim  = channels[-1]

    def forward(self, x):
        return self.features(x)
