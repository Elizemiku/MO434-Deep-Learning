# models/predictors.py
# Preditores do student + StudentModel completo

import torch.nn as nn

# Mapa de funções de ativação suportadas pelo preditor MLP
# ReLU: grad=1 para x>0, sem saturação, padrão em ConvNets
# GELU: x * Phi(x), suave e contínua, padrão em Transformers (GPT, BERT)
# SiLU: x * sigmoid(x) = Swish, usada em EfficientNet/MobileNetV3
#       gradiente mais rico que ReLU, útil quando o espaço de features
#       tem estrutura complexa de alta dimensão (ex: ResNet50 feat_dim=2048)
_ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


class PreditorPostGAP(nn.Module):
    """
    Mapeia features do student (apos GAP) para vetor C_teacher-dimensional.

    Tipo post-GAP:
      Alvo: vetor [B, C_t]
      Arquitetura: MLP com camada oculta
      Vantagem: mais simples, dimensão baixa

    O parametro "activation" permite trocar ReLU por GELU ou SiLU:
      - 'relu' (padrão): rápido, sem saturação, padrão histórico
      - 'gelu': gradiente suave (usado em BERT/GPT), pode ajudar
                em espaços de features de alta dimensão
      - 'silu': Swish — similar ao GELU, bom em redes de imagem
    """
    def __init__(self, dim_student: int, dim_teacher: int,
                 dim_hidden: int = 512, activation: str = 'relu'):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation.lower(), nn.ReLU)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_student, dim_hidden),
            act_cls(),
            # projeta para espaço do teacher
            nn.Linear(dim_hidden, dim_teacher),
        )

    def forward(self, features_student):
        # features_student: [B, C_s, H, W] -> predicao: [B, C_teacher]
        x = self.gap(features_student).flatten(1)
        return self.mlp(x)


class PreditorPreGAP(nn.Module):
    """
    Mapeia as features do student para o mapa espacial do teacher [B, C_teacher, 7, 7].

    Tipo pre-GAP:
      Alvo: mapa [B, C_t, 7, 7]
      Arquitetura: Conv 1x1 para expandir canais + adaptive pool
      Vantagem: preserva localização espacial
    """
    def __init__(self, dim_student: int, dim_teacher: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(dim_student, dim_teacher, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_teacher),
            nn.ReLU(),
        )
        # força a saída para 7x7 (igual ao teacher)
        self.pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, features_student):
        # features_student: [B, C_s, H, W] -> predicao: [B, C_teacher, 7, 7]
        x = self.expand(features_student)
        return self.pool(x)


class StudentModel(nn.Module):
    """
    Modelo student completo: encoder + preditor.

    O student deve ser consideravelmente mais leve que o teacher:
      - Menos de 10% dos GFLOPs do teacher
      - Menos de 20% dos parâmetros do teacher
    """
    def __init__(self, encoder: nn.Module, preditor: nn.Module):
        super().__init__()
        self.encoder  = encoder
        self.preditor = preditor

    def forward(self, x):
        features = self.encoder(x)
        return self.preditor(features)
