# kd_utils.py
# Utilitarios para Knowledge Distillation - MO434 Aprendizado Profundo para Visao Computacional
#
# Classes organizadas para manter o notebook limpo e focado na analise de resultados.
#
# Estrutura:
#   1. ImageTransforms   - transformacoes, normalizacao e filtros de imagem
#   2. DatasetManager    - carregamento e gestao dos datasets
#   3. conv_block        - bloco de convolucao (filtros e mascaras)
#   4. ResBlock          - bloco residual (skip connection)
#   5. PlainCNNEncoder   - encoder student baseline
#   6. DepthwiseCNNEncoder - encoder student leve (depthwise-separavel)
#   7. MiniResNetEncoder - encoder student com skip connections
#   8. TeacherWrapper    - encapsula qualquer backbone pre-treinado
#   9. PreditorPostGAP   - preditor de features post-GAP (Q2)
#  10. PreditorPreGAP    - preditor de features pre-GAP  (Q2)
#  11. StudentModel      - encoder + preditor completo
#  12. PerdaKD           - perda combinada MSE + CE (Q4)
#  13. PerdaRKD          - Relational Knowledge Distillation (Q5)
#  14. Trainer           - loops de treino das 3 fases (com deteccao de overfitting,
#                          early stopping por paciencia e rastreamento de gap)
#  15. Evaluator         - avaliacao final e metricas de eficiencia
#  16. plotar_curvas (loss + acuracia + gap de generalizacao com marcacao da melhor epoca),
#      plotar_comparacao_mse_rkd, plotar_graficos_analise

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models


# =============================================================================
# 1. TRANSFORMACOES DE IMAGEM
#    Pre-processamento para modelos pre-treinados no ImageNet
# =============================================================================

class ImageTransforms:
    """
    Transformacoes, normalizacao e filtros de imagem para modelos pre-treinados.

    Todos os backbones (VGG, ResNet, ConvNeXt) foram treinados com imagens
    normalizadas com a media e desvio padrao do ImageNet. E obrigatorio usar
    os mesmos valores:

        media  = [0.485, 0.456, 0.406]
        desvio = [0.229, 0.224, 0.225]

    Alem disso, os modelos esperam entradas de 224x224 pixels.
    """

    # parametros de normalizacao do ImageNet (obrigatorio para modelos pre-treinados)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    @staticmethod
    def get_train_transform():
        """
        Transformacoes com augmentacao para o conjunto de treino.
        A augmentacao aumenta a diversidade sem coletar novos dados.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(ImageTransforms.IMAGENET_MEAN, ImageTransforms.IMAGENET_STD),
        ])

    @staticmethod
    def get_val_transform():
        """
        Transformacoes padrao para validacao e teste (sem augmentacao).
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(ImageTransforms.IMAGENET_MEAN, ImageTransforms.IMAGENET_STD),
        ])

    @staticmethod
    def denormalize(tensor):
        """
        Inverte a normalizacao do ImageNet para visualizacao de imagens.
        Necessario para exibir imagens sem o artefato da normalizacao.
        Uso: img = ImageTransforms.denormalize(img_tensor)
        """
        mean_t = torch.tensor(ImageTransforms.IMAGENET_MEAN).view(3, 1, 1)
        std_t  = torch.tensor(ImageTransforms.IMAGENET_STD).view(3, 1, 1)
        return (tensor * std_t + mean_t).clamp(0, 1)


# =============================================================================
# 2. GESTAO DE DATASETS
# =============================================================================

class DatasetManager:
    """
    Carregamento e gestao dos datasets do projeto.

    Por que usar 2 datasets?
    O enunciado exige multiplos datasets para garantir que os resultados sejam
    generalizaveis e nao especificos de um unico dominio.

    Datasets disponibilizados:
    - flowers102: 102 classes de flores, ~8.000 imagens em resolucao variada
      splits oficiais: train (1020), val (1020), test (6149)
      ideal para Q1: fine-grained, ConvNeXt tende a se sair melhor

    - pets: Oxford-IIIT-Pet, 37 racas de caes e gatos, ~7.000 imagens
      testa transferencia em classificacao de granularidade fina com menos dados
    """

    def __init__(self, data_root='./data', batch_size=32, num_workers=2):
        self.data_root   = data_root
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.datasets    = {}

        # transformacoes reutilizadas por todos os datasets
        self.transform_train = ImageTransforms.get_train_transform()
        self.transform_val   = ImageTransforms.get_val_transform()

    def load_flowers102(self):
        """
        Carrega Flowers-102 com splits oficiais: train (1020), val (1020), test (6149).
        102 classes de flores em resolucao variada.
        Ideal para Q1: fine-grained, a diferenca entre teachers fica mais evidente.
        """
        train_raw = datasets.Flowers102(
            root=self.data_root, split='train', download=True,
            transform=self.transform_train)
        val_raw = datasets.Flowers102(
            root=self.data_root, split='val', download=True,
            transform=self.transform_val)
        test_raw = datasets.Flowers102(
            root=self.data_root, split='test', download=True,
            transform=self.transform_val)

        self.datasets['flowers102'] = {
            'n_classes': 102,
            'train': DataLoader(train_raw, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True),
            'val':   DataLoader(val_raw,   batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
            'test':  DataLoader(test_raw,  batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
        }
        print(f"flowers-102 -> treino: {len(train_raw):,} | val: {len(val_raw):,} | teste: {len(test_raw):,}")
        return self.datasets['flowers102']

    def load_oxford_pets(self):
        """
        Carrega Oxford-IIIT-Pet com 37 classes (racas de caes e gatos), ~7.000 imagens.
        Divisao: 80% treino, 20% validacao do conjunto trainval.
        Confirma que os resultados generalizam para um dominio diferente do Flowers-102.
        """
        trainval = datasets.OxfordIIITPet(
            root=self.data_root, split='trainval', download=True,
            transform=self.transform_train)
        test_raw = datasets.OxfordIIITPet(
            root=self.data_root, split='test', download=True,
            transform=self.transform_val)

        # divisao: 80% treino, 20% validacao do conjunto trainval
        n_train = int(0.8 * len(trainval))
        n_val   = len(trainval) - n_train
        tr, vl  = random_split(trainval, [n_train, n_val])

        self.datasets['pets'] = {
            'n_classes': 37,
            'train': DataLoader(tr,       batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True),
            'val':   DataLoader(vl,       batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
            'test':  DataLoader(test_raw, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True),
        }
        print(f"oxford-pets -> treino: {len(tr):,} | val: {len(vl):,} | teste: {len(test_raw):,}")
        return self.datasets['pets']

    def load_all(self):
        """Carrega todos os datasets e retorna o mapa de configuracoes."""
        self.load_flowers102()
        self.load_oxford_pets()
        return self.datasets

    def get(self, name):
        """Retorna configuracao de um dataset pelo nome ('flowers102' ou 'pets')."""
        if name not in self.datasets:
            raise KeyError(f"dataset '{name}' nao carregado. chame load_{name}() primeiro.")
        return self.datasets[name]

    def visualizar_amostras(self, n_imgs=8):
        """
        Visualizacao: amostras de imagens dos datasets carregados.
        Inverte a normalizacao do ImageNet para exibir as imagens corretamente.
        """
        nomes = list(self.datasets.keys())
        fig, axes = plt.subplots(len(nomes), n_imgs, figsize=(n_imgs * 2, len(nomes) * 2.5))
        if len(nomes) == 1:
            axes = [axes]

        for row, nome in enumerate(nomes):
            loader = self.datasets[nome]['val']
            imgs, labels = next(iter(loader))
            for i in range(min(n_imgs, len(imgs))):
                img = ImageTransforms.denormalize(imgs[i]).permute(1, 2, 0).numpy()
                axes[row][i].imshow(img)
                axes[row][i].set_title(f"cls {labels[i].item()}", fontsize=8)
                axes[row][i].axis('off')
            axes[row][0].set_ylabel(nome, fontsize=10, rotation=90)

        plt.suptitle("Amostras dos datasets (normalizacao invertida para visualizacao)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


# =============================================================================
# 3. BLOCOS DE CONVOLUCAO (FILTROS E MASCARAS)
# =============================================================================

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


# =============================================================================
# 4. ENCODERS DO STUDENT (Q3)
#    Tres variantes: Plain CNN, Depthwise, MiniResNet
# =============================================================================

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


# =============================================================================
# 5. TEACHER WRAPPER
# =============================================================================

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
        [CLASSIFICADOR TREINAVEL]   <- so esta parte e treinada na Fase 1
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
            # remove gap e fc originais, mantemos so as conv layers
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
            raise ValueError(f"backbone '{name}' nao suportado. use: vgg16, resnet50, convnext_small")

    def freeze_encoder(self):
        """
        Congela todos os parametros do encoder.
        Chamada obrigatoria antes da Fase 1: o encoder ja sabe extrair features
        (pre-treinado no ImageNet), apenas o classificador precisa de ajuste.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f"encoder {self.backbone_name} congelado.")

    def get_pre_gap(self, x):
        """Retorna feature map espacial antes do GAP: [B, C, 7, 7]"""
        return self.encoder(x)

    def get_post_gap(self, x):
        """Retorna vetor comprimido apos o GAP: [B, C]"""
        feat = self.encoder(x)
        return self.gap(feat).flatten(1)

    def forward(self, x):
        """Forward completo: encoder -> gap -> classificador -> logits"""
        feat   = self.encoder(x)
        pooled = self.gap(feat).flatten(1)
        return self.classifier(pooled)


# =============================================================================
# 6. PREDITORES DO STUDENT (Q2)
#    Q2 - Student deve prever pre-GAP ou post-GAP?
# =============================================================================

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


# =============================================================================
# 7. FUNCOES DE PERDA NAO-LINEARES (Q4 e Q5)
# =============================================================================

class PerdaKD(nn.Module):
    """
    Perda combinada MSE + Cross-Entropy para Knowledge Distillation (Q4).

    Formula:
        L_total = alpha * L_MSE + (1 - alpha) * L_CE

      - L_MSE: alinha features do student com features do teacher (distilacao)
      - L_CE:  mantém previsoes alinhadas com rotulos verdadeiros (supervisao)
      - alpha: hiperparametro para balancear os dois objetivos

    Valores de alpha:
      1.0 -> so MSE (destilacao pura, sem rotulos)
      0.0 -> so CE  (ignora teacher, treino supervisionado puro)
      0.7 -> ponto de partida recomendado
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_student, feat_teacher, logits_via_classifier, labels):
        """
        pred_student:          saida do preditor do student
        feat_teacher:          features do teacher (alvo)
        logits_via_classifier: predicao passada pelo classificador do teacher
        labels:                rotulos verdadeiros
        """
        l_mse = F.mse_loss(pred_student, feat_teacher)
        l_ce  = F.cross_entropy(logits_via_classifier, labels)
        perda = self.alpha * l_mse + (1 - self.alpha) * l_ce
        return perda, l_mse.item(), l_ce.item()


class PerdaRKD(nn.Module):
    """
    Relational Knowledge Distillation - RKD (Park et al., CVPR 2019).

    Ao inves de alinhar valores absolutos de features (como o MSE faz),
    alinha as relacoes entre amostras (distancias par-a-par):

    > "Amostras similares no espaco do teacher devem ser similares
    >  no espaco do student"

    Isso e invariante a escala das features e pode ser mais robusto quando
    o student tem capacidade menor que o teacher (capacity mismatch).

    Referencia: https://arxiv.org/abs/1904.05068
    """
    def __init__(self, weight_dist=1.0, weight_angle=2.0):
        super().__init__()
        self.w_dist  = weight_dist
        self.w_angle = weight_angle

    def forward(self, feat_student, feat_teacher):
        """feat_student e feat_teacher: [B, D] (features pos-gap)"""
        l_dist  = self._distancia(feat_student, feat_teacher)
        l_angle = self._angulo(feat_student, feat_teacher)
        return self.w_dist * l_dist + self.w_angle * l_angle

    def _distancia(self, fs, ft):
        """Normaliza distancias par-a-par e compara (invariante a escala)."""
        d_s = torch.cdist(fs, fs)
        d_t = torch.cdist(ft, ft)
        d_s = d_s / (d_s.mean() + 1e-8)  # normaliza pela media
        d_t = d_t / (d_t.mean() + 1e-8)
        return F.huber_loss(d_s, d_t)     # huber e robusto a outliers

    def _angulo(self, fs, ft):
        """Compara similaridade coseno par-a-par (geometria angular)."""
        fs_norm = F.normalize(fs, dim=1)  # projeta na esfera unitaria
        ft_norm = F.normalize(ft, dim=1)
        sim_s = fs_norm @ fs_norm.T       # similaridade coseno: [B, B]
        sim_t = ft_norm @ ft_norm.T
        return F.mse_loss(sim_s, sim_t)


# =============================================================================
# 8. TRAINER - LOOPS DE TREINO DAS 3 FASES
# =============================================================================

class Trainer:
    """
    Gerencia loops de treino para as tres fases do projeto KD.

    Fases:
      Fase 1: Treinar o CLASSIFICADOR do teacher (encoder congelado)
      Fase 2: Treinar ENCODER + PREDITOR do student (imitacao de features)
      Fase 3: (avaliacao, ver Evaluator)
    """

    def __init__(self, device):
        self.device = device

    # ── helpers internos ──────────────────────────────────────────────────────

    def _treinar_batch(self, model, dados, optimizer):
        """
        Executa um passo de treino: forward -> loss -> backward -> atualizar pesos.
        Retorna (loss, acuracia) do batch.
        """
        model.train()
        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        optimizer.zero_grad()                    # 1. zera gradientes acumulados
        logits = model(imgs)                     # 2. forward pass
        loss   = F.cross_entropy(logits, labels) # 3. cross-entropy loss
        loss.backward()                          # 4. backpropagation
        optimizer.step()                         # 5. atualiza pesos

        acc = (logits.argmax(1) == labels).float().mean().item()
        return loss.item(), acc

    @torch.no_grad()
    def _validar_batch(self, model, dados):
        """
        Avalia o modelo sem computar gradientes (mais rapido e menos memoria).
        model.eval() desativa dropout e usa estatisticas fixas do batchnorm.
        """
        model.eval()
        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = model(imgs)
        loss   = F.cross_entropy(logits, labels)
        acc    = (logits.argmax(1) == labels).float().mean().item()
        return loss.item(), acc

    @torch.no_grad()
    def avaliar_loader(self, model, loader):
        """Avalia o modelo em um DataLoader completo, retornando acuracia media."""
        accs = []
        for dados in loader:
            _, a = self._validar_batch(model, dados)
            accs.append(a)
        return np.mean(accs)

    # ── fase 1: treinar classificador do teacher ──────────────────────────────

    def treinar_fase1(self, model, loader_train, loader_val, optimizer, scheduler,
                      n_epochs, descricao="",
                      patience=5, overfitting_threshold=0.15):
        """
        Loop de treino completo com early stopping e deteccao de overfitting.

        Na Fase 1:
          1. O encoder esta congelado (nenhum gradiente passa por ele)
          2. Apenas o classificador e atualizado com Cross-Entropy Loss

        Por que isso e critico para o KD?
        O classificador treinado aqui e reutilizado exatamente na Fase 3 para
        avaliar o student. Isso forca o student a produzir features que estejam
        no mesmo espaco do teacher.

        Deteccao de overfitting:
          - gap_acc = acc_tr - acc_vl: mede distancia entre treino e validacao.
            Quando gap_acc > overfitting_threshold por multiplas epocas, o modelo
            memorizou o treino e nao generaliza.
          - val_loss subindo enquanto train_loss cai: sinal classico de overfitting.
          - early stopping: para quando val_acc nao melhora por `patience` epocas,
            restaurando os melhores pesos automaticamente.

        Parametros:
          patience:               epocas sem melhora antes de parar (0 = desativado)
          overfitting_threshold:  gap acc_tr - acc_vl que dispara aviso (padrao 0.15)
        """
        historico    = defaultdict(list)
        melhor_acc   = 0.0
        melhor_pesos = None
        melhor_epoch = 1
        sem_melhora  = 0  # contador de epocas sem melhora (early stopping)

        for epoch in range(1, n_epochs + 1):
            # --- treino ---
            losses_tr, accs_tr = [], []
            for dados in tqdm(loader_train, desc=f"[{descricao}] epoca {epoch}/{n_epochs}", leave=False):
                l, a = self._treinar_batch(model, dados, optimizer)
                losses_tr.append(l); accs_tr.append(a)

            # --- validacao ---
            losses_vl, accs_vl = [], []
            for dados in loader_val:
                l, a = self._validar_batch(model, dados)
                losses_vl.append(l); accs_vl.append(a)

            loss_tr = np.mean(losses_tr);  acc_tr = np.mean(accs_tr)
            loss_vl = np.mean(losses_vl);  acc_vl = np.mean(accs_vl)
            # gap de generalizacao: positivo indica que o modelo vai melhor no treino
            # do que na validacao. Valores altos (> overfitting_threshold) sinalizam overfitting
            gap_acc = acc_tr - acc_vl

            historico['loss_tr'].append(loss_tr);  historico['acc_tr'].append(acc_tr)
            historico['loss_vl'].append(loss_vl);  historico['acc_vl'].append(acc_vl)
            historico['gap_acc'].append(gap_acc)   # rastreia gap de generalizacao

            if scheduler:
                scheduler.step()

            # --- deteccao de overfitting ---
            # sinal 1: gap de acuracia alto e persistente
            # sinal 2: val_loss subindo enquanto train_loss cai (divergencia classica)
            avisos = []
            if gap_acc > overfitting_threshold:
                avisos.append(f"gap={gap_acc:.3f}>{overfitting_threshold}")
            if (len(historico['loss_vl']) >= 3 and
                    historico['loss_vl'][-1] > historico['loss_vl'][-2] > historico['loss_vl'][-3] and
                    historico['loss_tr'][-1] < historico['loss_tr'][-2]):
                avisos.append("val_loss subindo 3x seguidas")

            # --- early stopping ---
            if acc_vl > melhor_acc:
                melhor_acc   = acc_vl
                melhor_pesos = deepcopy(model.state_dict())
                melhor_epoch = epoch
                sem_melhora  = 0
            else:
                sem_melhora += 1

            if epoch % 5 == 0 or epoch == 1 or avisos:
                status = (f"  epoca {epoch:3d} | loss_tr={loss_tr:.4f} acc_tr={acc_tr:.3f} | "
                          f"loss_vl={loss_vl:.4f} acc_vl={acc_vl:.3f} | gap={gap_acc:+.3f}")
                if avisos:
                    status += f"  [OVERFITTING: {'; '.join(avisos)}]"
                print(status)

            if patience and sem_melhora >= patience:
                print(f"  early stopping na epoca {epoch} "
                      f"(sem melhora por {patience} epocas). melhor: epoca {melhor_epoch}")
                break

        # restaura melhores pesos
        if melhor_pesos:
            model.load_state_dict(melhor_pesos)
        historico['melhor_epoch'] = [melhor_epoch]  # lista para compatibilidade com defaultdict
        print(f"  melhor acc validacao: {melhor_acc:.4f} (epoca {melhor_epoch})")
        return historico

    # ── fase 2: destilacao ────────────────────────────────────────────────────

    def _treinar_batch_kd(self, student, teacher, dados, optimizer, perda_fn,
                          modo_target='post_gap'):
        """
        Um passo de treino de destilacao:
          1. Extrai features do teacher (sem gradiente)
          2. Student prediz essas features
          3. Calcula perda combinada MSE + CE
          4. Backpropaga e atualiza apenas pesos do student

        Ponto critico: NUNCA deixe gradientes fluirem pelo encoder do teacher.
        torch.no_grad() e essencial para economizar memoria e tempo de computo.
        """
        student.train()
        teacher.eval()  # teacher sempre em eval (batchnorm usa estatisticas fixas)

        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        # extrai targets do teacher sem gradiente (economiza memoria)
        with torch.no_grad():
            if modo_target == 'post_gap':
                feat_teacher = teacher.get_post_gap(imgs)   # [B, C_t]
            else:
                feat_teacher = teacher.get_pre_gap(imgs)    # [B, C_t, 7, 7]

        optimizer.zero_grad()
        pred_student = student(imgs)   # [B, C_t] ou [B, C_t, 7, 7]

        # passa predicao pelo classificador do teacher para obter logits
        with torch.no_grad():
            if modo_target == 'post_gap':
                logits = teacher.classifier(pred_student)
            else:
                pooled = teacher.gap(pred_student).flatten(1)
                logits = teacher.classifier(pooled)

        perda, l_mse, l_ce = perda_fn(pred_student, feat_teacher, logits, labels)
        perda.backward()
        # clipa gradientes para evitar explosao (importante para redes profundas)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        acc = (logits.argmax(1) == labels).float().mean().item()
        return perda.item(), l_mse, l_ce, acc

    @torch.no_grad()
    def validar_student(self, student, teacher, loader_val, modo_target='post_gap'):
        """Avalia o student usando o classificador do teacher (preparacao para fase 3)."""
        student.eval(); teacher.eval()
        accs = []
        for imgs, labels in loader_val:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            pred = student(imgs)
            if modo_target == 'post_gap':
                logits = teacher.classifier(pred)
            else:
                pooled = teacher.gap(pred).flatten(1)
                logits = teacher.classifier(pooled)
            accs.append((logits.argmax(1) == labels).float().mean().item())
        return np.mean(accs)

    @torch.no_grad()
    def _validar_student_kd(self, student, teacher, loader_val, perda_fn,
                            modo_target='post_gap'):
        """
        Avalia o student retornando (acc_vl, loss_vl) usando a perda de destilacao.
        Necessario para detectar divergencia de val_loss na Fase 2 (sinal de overfitting).
        """
        student.eval(); teacher.eval()
        accs, perdas = [], []
        for imgs, labels in loader_val:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            feat_teacher = (teacher.get_post_gap(imgs) if modo_target == 'post_gap'
                            else teacher.get_pre_gap(imgs))
            pred = student(imgs)
            if modo_target == 'post_gap':
                logits = teacher.classifier(pred)
            else:
                pooled = teacher.gap(pred).flatten(1)
                logits = teacher.classifier(pooled)
            p, _, _ = perda_fn(pred, feat_teacher, logits, labels)
            accs.append((logits.argmax(1) == labels).float().mean().item())
            perdas.append(p.item())
        return np.mean(accs), np.mean(perdas)

    def treinar_fase2(self, student, teacher, loader_train, loader_val, perda_fn,
                      n_epochs, lr, modo_target='post_gap', descricao='',
                      patience=5, overfitting_threshold=0.15):
        """
        Loop completo de destilacao com registro de metricas e deteccao de overfitting.

        O teacher esta completamente congelado. O student aprende a imitar
        as features do teacher atraves do gradiente da perda de destilacao.

        Fluxo:
          Imagem -> [TEACHER ENCODER, congelado] -> features_teacher (alvo, sem grad)
          Imagem -> [STUDENT ENCODER, treina]    -> features_student
                 -> [PREDITOR, treina]           -> pred_features
                                                       |
                                            MSE(pred_features, features_teacher)
                                            + CE(classifier(pred_features), labels)

        Deteccao de overfitting:
          - gap_acc = acc_tr - acc_vl: distancia entre treino e validacao.
          - val_loss subindo enquanto train_loss cai: divergencia classica.
          - early stopping por paciencia: para quando val_acc estagna.

        Parametros:
          patience:               epocas sem melhora antes de parar (0 = desativado)
          overfitting_threshold:  gap acc_tr - acc_vl que dispara aviso (padrao 0.15)
        """
        optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        historico    = defaultdict(list)
        melhor_acc   = 0.0
        melhor_pesos = None
        melhor_epoch = 1
        sem_melhora  = 0

        for epoch in range(1, n_epochs + 1):
            perdas, mselist, celist, accs_tr = [], [], [], []
            for dados in tqdm(loader_train, desc=f"[{descricao}] ep.{epoch}", leave=False):
                p, m, c, a = self._treinar_batch_kd(
                    student, teacher, dados, optimizer, perda_fn, modo_target)
                perdas.append(p); mselist.append(m); celist.append(c); accs_tr.append(a)

            # avalia val com perda para monitorar divergencia de loss
            acc_vl, loss_vl = self._validar_student_kd(
                student, teacher, loader_val, perda_fn, modo_target)
            scheduler.step()

            acc_tr  = np.mean(accs_tr)
            loss_tr = np.mean(perdas)
            # gap de generalizacao: positivo indica melhor desempenho no treino do que na val
            gap_acc = acc_tr - acc_vl

            historico['perda'].append(loss_tr)
            historico['loss_vl'].append(loss_vl)
            historico['mse'].append(np.mean(mselist))
            historico['acc_tr'].append(acc_tr)
            historico['acc_vl'].append(acc_vl)
            historico['gap_acc'].append(gap_acc)  # rastreia gap de generalizacao

            # --- deteccao de overfitting ---
            avisos = []
            if gap_acc > overfitting_threshold:
                avisos.append(f"gap={gap_acc:.3f}>{overfitting_threshold}")
            if (len(historico['loss_vl']) >= 3 and
                    historico['loss_vl'][-1] > historico['loss_vl'][-2] > historico['loss_vl'][-3] and
                    historico['perda'][-1] < historico['perda'][-2]):
                avisos.append("val_loss subindo 3x seguidas")

            # --- early stopping ---
            if acc_vl > melhor_acc:
                melhor_acc   = acc_vl
                melhor_pesos = deepcopy(student.state_dict())
                melhor_epoch = epoch
                sem_melhora  = 0
            else:
                sem_melhora += 1

            if epoch % 5 == 0 or epoch == 1 or avisos:
                status = (f"  ep.{epoch:3d} | mse={np.mean(mselist):.4f} | "
                          f"acc_tr={acc_tr:.3f} acc_vl={acc_vl:.3f} | gap={gap_acc:+.3f}")
                if avisos:
                    status += f"  [OVERFITTING: {'; '.join(avisos)}]"
                print(status)

            if patience and sem_melhora >= patience:
                print(f"  early stopping na epoca {epoch} "
                      f"(sem melhora por {patience} epocas). melhor: epoca {melhor_epoch}")
                break

        if melhor_pesos:
            student.load_state_dict(melhor_pesos)
        historico['melhor_epoch'] = [melhor_epoch]
        print(f"  melhor acc validacao: {melhor_acc:.4f} (epoca {melhor_epoch})")
        return historico, melhor_acc

    # ── Q5: comparacao mse vs rkd ─────────────────────────────────────────────

    def _treinar_batch_rkd(self, student, teacher, dados, optimizer, perda_rkd_fn,
                           alpha_ce=0.3):
        """
        Passo de treino com RKD:
        l_total = RKD(feat_student, feat_teacher) + alpha_ce * CE(logits, labels)
        """
        student.train(); teacher.eval()
        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        with torch.no_grad():
            feat_teacher = teacher.get_post_gap(imgs)

        optimizer.zero_grad()
        pred_student = student(imgs)

        # rkd compara relacoes par-a-par no espaco de features
        l_rkd = perda_rkd_fn(pred_student, feat_teacher)

        # ce supervisionado para nao perder alinhamento com rotulos
        with torch.no_grad():
            logits = teacher.classifier(pred_student)
        l_ce  = F.cross_entropy(logits, labels)

        perda = l_rkd + alpha_ce * l_ce
        perda.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        acc = (logits.argmax(1) == labels).float().mean().item()
        return perda.item(), l_rkd.item(), l_ce.item(), acc

    def comparar_mse_vs_rkd(self, teacher, datasets, teacher_nome, dataset_nome,
                             n_epochs=20, lr=1e-3, student_channels=(32, 64, 128, 256)):
        """
        Compara MSE baseline vs RKD com mesmo encoder e teacher (Q5).

        MSE forca o student a copiar os valores absolutos das features.
        RKD alinha as relacoes entre amostras, invariante a escala.
        Util quando o student tem capacidade menor (capacity mismatch).

        Retorna dict com accs_vl e melhor_acc para cada metodo.
        """
        resultados_comp = {}

        for nome_perda, modo in [('MSE_baseline', 'mse'), ('RKD', 'rkd')]:
            print(f"  treinando: {nome_perda}")

            enc    = PlainCNNEncoder(student_channels)
            pred   = PreditorPostGAP(enc.out_dim, teacher.feat_dim)
            stu    = StudentModel(enc, pred).to(self.device)
            optim_ = optim.AdamW(stu.parameters(), lr=lr, weight_decay=1e-4)
            sched_ = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=n_epochs)

            perda_fn_mse = PerdaKD(alpha=0.7)
            perda_fn_rkd = PerdaRKD()
            accs_vl = []

            for epoch in range(1, n_epochs + 1):
                for dados in datasets[dataset_nome]['train']:
                    if modo == 'mse':
                        self._treinar_batch_kd(stu, teacher, dados, optim_, perda_fn_mse)
                    else:
                        self._treinar_batch_rkd(stu, teacher, dados, optim_, perda_fn_rkd)

                acc = self.validar_student(stu, teacher, datasets[dataset_nome]['val'])
                accs_vl.append(acc)
                sched_.step()

            resultados_comp[nome_perda] = {'accs_vl': accs_vl, 'melhor_acc': max(accs_vl)}
            print(f"    melhor acc: {max(accs_vl):.4f}")

        return resultados_comp


# =============================================================================
# 9. EVALUATOR - AVALIACAO FINAL E METRICAS DE EFICIENCIA
# =============================================================================

class Evaluator:
    """
    Avaliacao final do student (Fase 3) e calculo de metricas de eficiencia.
    """

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def avaliar_fase3(self, student, teacher, loader_test, modo_target='post_gap'):
        """
        Avaliacao definitiva (Fase 3): usa classificador do teacher para avaliar student.

        O classificador nunca viu as features do student durante o treinamento.
        Se o student aprendeu bem as representacoes, o classificador do teacher
        conseguira classificar as predicoes do student com alta acuracia
        (transferencia real de conhecimento).

        Retorna (top1_acc, top5_acc).
        """
        student.eval(); teacher.eval()
        accs, top5_accs = [], []

        for imgs, labels in loader_test:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            pred = student(imgs)

            if modo_target == 'post_gap':
                logits = teacher.classifier(pred)
            else:
                pooled = teacher.gap(pred).flatten(1)
                logits = teacher.classifier(pooled)

            # top-1
            accs.append((logits.argmax(1) == labels).float().mean().item())

            # top-5 (mais informativo para datasets com muitas classes)
            if logits.shape[1] >= 5:
                top5 = logits.topk(5, dim=1).indices
                top5_accs.append((top5 == labels.unsqueeze(1)).any(1).float().mean().item())

        return np.mean(accs), (np.mean(top5_accs) if top5_accs else None)

    def contar_eficiencia(self, model, input_size=(1, 3, 224, 224)):
        """
        Retorna (gflops, n_parametros_M) de um modelo.
        Requer: pip install fvcore
        """
        model.eval()
        dummy = torch.randn(*input_size)
        try:
            from fvcore.nn import FlopCountAnalysis
            flops  = FlopCountAnalysis(model, dummy)
            gflops = flops.total() / 1e9
        except (ImportError, Exception):
            gflops = None
        params = sum(p.numel() for p in model.parameters()) / 1e6
        return gflops, params

    def relatorio_eficiencia(self, teacher, resultados_fase2,
                             filtro_alpha=0.7, filtro_target='post_gap'):
        """
        Imprime tabela de eficiencia: GFLOPs e parametros do teacher vs students.
        """
        print("Eficiencia Computacional\n")
        print(f"{'modelo':30s} {'gflops':>8s} {'params_M':>10s} {'reducao':>12s}")
        print("─" * 65)

        t_cpu = teacher.cpu()
        gf_t, pr_t = self.contar_eficiencia(t_cpu)
        gf_str = f"{gf_t:.3f}" if gf_t else "n/d"
        print(f"  {teacher.backbone_name + ' (teacher)':28s} {gf_str:>8s} {pr_t:>10.1f}M {'(ref)':>12s}")

        for exp in resultados_fase2:
            if exp['alpha'] == filtro_alpha and exp['modo_target'] == filtro_target:
                stu_cpu = exp['student_obj'].cpu()
                gf_s, pr_s = self.contar_eficiencia(stu_cpu)
                reducao    = f"{pr_s/pr_t*100:.1f}%"
                gf_str     = f"{gf_s:.3f}" if gf_s else "n/d"
                print(f"  {exp['encoder']:28s} {gf_str:>8s} {pr_s:>10.2f}M {reducao:>12s}")


# =============================================================================
# 10. VISUALIZACAO
# =============================================================================

def plotar_curvas(historico, titulo="", overfitting_threshold=0.15):
    """
    Plota curvas de loss, acuracia e gap de generalizacao.

    O terceiro grafico (gap) facilita a leitura do overfitting:
      - gap = acc_treino - acc_validacao
      - valores acima da linha vermelha tracejada (threshold) indicam overfitting
      - linha vertical cinza marca a melhor epoca (early stopping ou melhor val_acc)
    """
    tem_gap = 'gap_acc' in historico and len(historico['gap_acc']) > 0
    n_plots = 3 if tem_gap else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 1, 4))
    if n_plots == 2:
        axes = list(axes)

    # marca a melhor epoca, se disponivel
    melhor_epoch = None
    if 'melhor_epoch' in historico and historico['melhor_epoch']:
        melhor_epoch = historico['melhor_epoch'][0] - 1  # indice 0-based

    def _marcar_melhor(ax):
        if melhor_epoch is not None:
            ax.axvline(melhor_epoch, color='gray', linestyle='--', linewidth=1,
                       alpha=0.7, label=f'melhor (ep.{melhor_epoch + 1})')
            # sombreia a regiao apos a melhor epoca como zona de possivel overfitting
            if melhor_epoch < len(historico['acc_vl']) - 1:
                ax.axvspan(melhor_epoch, len(historico['acc_vl']) - 1,
                           alpha=0.07, color='red', label='zona apos melhor epoca')

    # --- grafico 1: loss ---
    ax = axes[0]
    epocas = range(len(historico['loss_tr']))
    ax.plot(epocas, historico['loss_tr'], label='treino')
    key_lv = 'loss_vl' if 'loss_vl' in historico else 'perda'
    if key_lv in historico and len(historico[key_lv]) == len(historico['loss_tr']):
        ax.plot(epocas, historico[key_lv], label='validacao')
    _marcar_melhor(ax)
    ax.set_title(f'loss - {titulo}'); ax.set_xlabel('epoca')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- grafico 2: acuracia ---
    ax = axes[1]
    epocas = range(len(historico['acc_tr']))
    ax.plot(epocas, historico['acc_tr'], label='treino')
    ax.plot(epocas, historico['acc_vl'], label='validacao')
    _marcar_melhor(ax)
    ax.set_title(f'acuracia - {titulo}'); ax.set_xlabel('epoca')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- grafico 3: gap de generalizacao (acc_tr - acc_vl) ---
    if tem_gap:
        ax = axes[2]
        epocas_gap = range(len(historico['gap_acc']))
        ax.plot(epocas_gap, historico['gap_acc'], color='purple', label='gap (tr - val)')
        ax.fill_between(epocas_gap, historico['gap_acc'], alpha=0.15, color='purple')
        # linha de threshold: acima = risco de overfitting
        ax.axhline(overfitting_threshold, color='red', linestyle='--', linewidth=1,
                   label=f'threshold ({overfitting_threshold})')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
        _marcar_melhor(ax)
        ax.set_title(f'gap de generalizacao - {titulo}')
        ax.set_xlabel('epoca'); ax.set_ylabel('acc_treino - acc_val')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()


def plotar_comparacao_mse_rkd(resultados_comp, teacher_nome, dataset_nome):
    """Plota comparacao de curvas de validacao entre MSE baseline e RKD."""
    plt.figure(figsize=(8, 4))
    for nome, res in resultados_comp.items():
        plt.plot(res['accs_vl'], label=f"{nome} (melhor={res['melhor_acc']:.3f})")
    plt.xlabel('epoca'); plt.ylabel('acuracia validacao')
    plt.title(f'MSE vs RKD | teacher={teacher_nome}, dataset={dataset_nome}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()


def plotar_graficos_analise(df_resultados):
    """
    Graficos de analise final respondendo Q1, Q3 e Q4:
      - Q1: comparacao de teachers
      - Q4: ablacao do peso alpha
      - Q3: trade-off acuracia vs parametros
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Q1: comparacao de teachers
    ax = axes[0]
    dados_q1 = df_resultados[
        (df_resultados['encoder'] == 'PlainCNNEncoder') &
        (df_resultados['target']  == 'post_gap') &
        (df_resultados['alpha']   == 0.7)
    ].sort_values('acc_test', ascending=True)
    ax.barh(dados_q1['teacher'], dados_q1['acc_test'], color='steelblue', alpha=0.8)
    ax.set_title('Q1 - Qual teacher transfere melhor?', fontsize=11)
    ax.set_xlabel('acuracia no teste'); ax.grid(True, alpha=0.3, axis='x')

    # Q4: ablacao de alpha
    ax = axes[1]
    dados_q4 = df_resultados[
        (df_resultados['teacher'] == 'resnet50') &
        (df_resultados['encoder'] == 'PlainCNNEncoder') &
        (df_resultados['target']  == 'post_gap')
    ].sort_values('alpha')
    ax.plot(dados_q4['alpha'], dados_q4['acc_test'], 'o-', color='darkorange', linewidth=2)
    ax.set_title('Q4 - Ablacao do peso alpha', fontsize=11)
    ax.set_xlabel('alpha (peso do MSE)'); ax.set_ylabel('acuracia no teste')
    ax.grid(True, alpha=0.3)

    # Q3: acuracia vs parametros
    ax = axes[2]
    dados_q3 = df_resultados[
        (df_resultados['teacher'] == 'resnet50') &
        (df_resultados['target']  == 'post_gap') &
        (df_resultados['alpha']   == 0.7)
    ]
    for _, row in dados_q3.iterrows():
        ax.scatter(row['params_M'], row['acc_test'], s=120, zorder=5)
        label = row['encoder'].replace('Encoder', '').replace('CNN', '')
        ax.annotate(label, (row['params_M'], row['acc_test']),
                    textcoords='offset points', xytext=(5, 3), fontsize=9)
    ax.set_title('Q3 - Trade-off: acuracia vs parametros', fontsize=11)
    ax.set_xlabel('parametros do student (M)'); ax.set_ylabel('acuracia no teste')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Analise dos Experimentos - Knowledge Distillation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
