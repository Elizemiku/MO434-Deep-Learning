# models/predictors.py
# Preditores do student + StudentModel completo - MO434 (Q2)
#
# Q2: O student deve prever features pre-GAP ou post-GAP do teacher?
#
#   PreditorPostGAP - alvo: vetor [B, C_teacher]       (simples, MLP)
#   PreditorPreGAP  - alvo: mapa  [B, C_teacher, 7, 7] (preserva espacialidade)
#   StudentModel    - encoder + preditor combinados

import torch.nn as nn


class PreditorPostGAP(nn.Module):
    """
    Mapeia features do student (apos GAP) para vetor C_teacher-dimensional.

    Tipo post-GAP:
      Alvo: vetor [B, C_t]
      Arquitetura: MLP com camada oculta
      Vantagem: simples, dimensao baixa
    """
    def __init__(self, dim_student: int, dim_teacher: int, dim_hidden: int = 512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_student, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_teacher),  # projeta para espaco do teacher
        )

    def forward(self, features_student):
        # features_student: [B, C_s, H, W] -> predicao: [B, C_teacher]
        x = self.gap(features_student).flatten(1)
        return self.mlp(x)


class PreditorPreGAP(nn.Module):
    """
    Mapeia features do student para mapa espacial do teacher [B, C_teacher, 7, 7].

    Tipo pre-GAP:
      Alvo: mapa [B, C_t, 7, 7]
      Arquitetura: Conv 1x1 para expandir canais + adaptive pool
      Vantagem: preserva localizacao espacial
    """
    def __init__(self, dim_student: int, dim_teacher: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(dim_student, dim_teacher, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_teacher),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(7)  # forca saida para 7x7 (igual ao teacher)

    def forward(self, features_student):
        # features_student: [B, C_s, H, W] -> predicao: [B, C_teacher, 7, 7]
        x = self.expand(features_student)
        return self.pool(x)


class StudentModel(nn.Module):
    """
    Modelo student completo: encoder + preditor.

    O student deve ser consideravelmente mais leve que o teacher:
      - Menos de 10% dos GFLOPs do teacher
      - Menos de 20% dos parametros do teacher
    """
    def __init__(self, encoder: nn.Module, preditor: nn.Module):
        super().__init__()
        self.encoder  = encoder
        self.preditor = preditor

    def forward(self, x):
        features = self.encoder(x)
        return self.preditor(features)
