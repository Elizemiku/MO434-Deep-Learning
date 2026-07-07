# models/blocks.py
# Blocos primitivos de convolução reutilizados pelos encoders

import torch.nn as nn


def conv_block(c_in, c_out, stride=1, dw=False):
    """
    Bloco de convolução com batchnorm e relu.

    Se dw=True: usa convolução depthwise-separável (muito mais eficiente).
    Convolução depthwise-separável: 1 filtro por canal + mistura de canais separada.
    Reduz operações em ~8x comparado a convolução padrão com mesma capacidade.
    """
    if dw:
        # depthwise separável: convolução por canal + pointwise para mistura de canais
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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., CVPR 2018).

    Recalibra a resposta de cada canal aplicando atenção global:
      1. Squeeze: GAP -> vetor [B, C] (contexto global de cada canal)
      2. Excitation: 2 camadas FC -> pesos de atenção por canal em (0,1)
      3. Scale: multiplica cada canal pelo seu peso

    Funciona como um filtro adaptativo sobre os mapas de features:
    canais mais discriminativos para a tarefa ganham maior peso.
    Adiciona apenas 2*C*C/r parâmetros extras (r=redução, padrão=16).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(-1, x.size(1), 1, 1)
        # reponde canais pelo mapa de atenção aprendido
        return x * w


class ResBlock(nn.Module):
    """
    Bloco residual: y = F(x) + x (skip connection).
    Melhora o fluxo de gradiente em redes profundas,
    permitindo treinar redes mais profundas sem degradação.
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
        # adiciona a entrada original (residual)
        return self.relu(self.conv(x) + x)
