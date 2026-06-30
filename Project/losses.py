# losses.py
# Funcoes de perda para Knowledge Distillation - MO434
#
#   PerdaKD   - perda combinada MSE + Cross-Entropy         
#   PerdaRKD  - Relational Knowledge Distillation 

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerdaKD(nn.Module):
    """
    Perda combinada MSE + Cross-Entropy para Knowledge Distillation.

    Formula:
        L_total = alpha * L_MSE + (1 - alpha) * L_CE

      - L_MSE: alinha features do student com features do teacher (destilacao)
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
    essa funcao alinha as relacoes entre amostras (distancias par-a-par):

    "Amostras similares no espaco do teacher devem ser similares no espaco do student"

    A escala das features e pode ser mais robusta quando
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
