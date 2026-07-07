# models/teacher.py
# TeacherWrapper - encapsula qualquer backbone pre-treinado

import torch.nn as nn
from torchvision import models


class TeacherWrapper(nn.Module):
    """
    Encapsula qualquer backbone pre-treinado expondo:
      encoder:    extrai feature maps (sem gap)
      gap:        global average pooling
      classifier: camada linear final

    Permite extrair pre-gap [B, C, 7, 7] e post-gap [B, C] de forma uniforme.

    Estrutura do backbone:
        Imagem [B, 3, 224, 224]
            |
        [ENCODER CONGELADO]   <- pesos ImageNet, nunca atualizados
            |
        Feature map pre-GAP: [B, C, 7, 7]
            |
        Global Average Pooling (GAP)
            |
        Vetor post-GAP: [B, C]
            |
        [CLASSIFICADOR TREINAVEL]   <- só esta parte é treinada na Fase 1
            |
        Logits: [B, n_classes]

    Teachers comparados no projeto:
        VGG-16:         138M params, 15.5 GFLOPs, dim post-GAP=512
        ResNet-50:       25M params,  4.1 GFLOPs, dim post-GAP=2048
        ConvNeXt-Small:  50M params,  8.7 GFLOPs, dim post-GAP=768
    """

    def __init__(self, backbone_name: str, n_classes: int):
        super().__init__()
        self.backbone_name = backbone_name
        self._build(backbone_name, n_classes)

    def _build(self, name, n_classes):
        if name == 'vgg16':
            base = models.vgg16(weights='IMAGENET1K_V1')
            self.encoder    = base.features           # saida: [B, 512, 7, 7]
            self.gap        = nn.AdaptiveAvgPool2d(1) # saida: [B, 512, 1, 1]
            self.feat_dim   = 512
            # substitui o classificador original por um adequado ao novo dataset
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, n_classes)
            )

        elif name == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V2')
            # remove gap e fc originais, mantemos só as conv layers
            self.encoder    = nn.Sequential(*list(base.children())[:-2])  # [B, 2048, 7, 7]
            self.gap        = nn.AdaptiveAvgPool2d(1)
            self.feat_dim   = 2048
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, n_classes)
            )

        elif name == 'convnext_small':
            base = models.convnext_small(weights='IMAGENET1K_V1')
            self.encoder    = base.features           # saida: [B, 768, 7, 7]
            self.gap        = nn.AdaptiveAvgPool2d(1)
            self.feat_dim   = 768
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(768),
                nn.Linear(768, n_classes)
            )
        else:
            raise ValueError(f"Backbone '{name}' não suportado. Usar: vgg16, resnet50, convnext_small")

    def freeze_encoder(self):
        """
        Congela todos os parâmetros do encoder.
        Chamada obrigatória antes da Fase 1: o encoder já sabe extrair features
        (pré-treinado no ImageNet), apenas o classificador precisa de ajuste.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f"Encoder {self.backbone_name} congelado.")

    def get_pre_gap(self, x):
        # Retorna feature map espacial antes do GAP: [B, C, 7, 7]
        return self.encoder(x)

    def get_post_gap(self, x):
        # Retorna vetor comprimido após o GAP: [B, C]
        feat = self.encoder(x)
        return self.gap(feat).flatten(1)

    def forward(self, x):
        # Forward completo: encoder -> gap -> classificador -> logits
        feat   = self.encoder(x)
        pooled = self.gap(feat).flatten(1)
        return self.classifier(pooled)
