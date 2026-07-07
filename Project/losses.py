# losses.py
# Funcoes de perda para o Knowledge Distillation

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerdaKD(nn.Module):
    """
    Perda combinada MSE + Cross-Entropy para Knowledge Distillation.

    Formula:
        L_total = alpha * L_MSE + (1 - alpha) * L_CE

      - L_MSE: Alinha features do student com features do teacher (destilação)
      - L_CE:  Mantém previsoes alinhadas com rótulos verdadeiros (supervisão)
      - alpha: Hiperparâmetro para balancear os dois objetivos

    Valores de alpha:
      1.0 -> só MSE (destilação pura, sem rótulos)
      0.0 -> só CE  (ignora teacher, treino supervisionado puro)
      0.7 -> valor inicial recomendado
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_student, feat_teacher, logits_via_classifier, labels):
        """
        pred_student:          Saida do preditor do student
        feat_teacher:          Features do teacher
        logits_via_classifier: Predição passada pelo classificador do teacher
        labels:                Rótulos verdadeiros
        """
        l_mse = F.mse_loss(pred_student, feat_teacher)
        l_ce  = F.cross_entropy(logits_via_classifier, labels)
        perda = self.alpha * l_mse + (1 - self.alpha) * l_ce
        return perda, l_mse.item(), l_ce.item()


class PerdaRKD(nn.Module):
    """
    Relational Knowledge Distillation - RKD (Park et al., CVPR 2019).

    Ao invés de alinhar valores absolutos de features (como o MSE faz),
    essa função alinha as relações entre amostras (distâncias par-a-par), pois:

    "Amostras similares no espaço do teacher devem ser similares no espaço do student"

    A escala das features pode ser mais robusta quando o student tem capacidade menor
    que o teacher (capacity mismatch).

    Referência: https://arxiv.org/abs/1904.05068
    """
    def __init__(self, weight_dist=1.0, weight_angle=2.0):
        super().__init__()
        self.w_dist  = weight_dist
        self.w_angle = weight_angle

    def forward(self, feat_student, feat_teacher):
        # feat_student e feat_teacher: [B, D] (features pós-gap)
        l_dist  = self._distancia(feat_student, feat_teacher)
        l_angle = self._angulo(feat_student, feat_teacher)
        return self.w_dist * l_dist + self.w_angle * l_angle

    def _distancia(self, fs, ft):
        # Normaliza as distancias par-a-par e faz a comparação (invariante a escala).
        d_s = torch.cdist(fs, fs)
        d_t = torch.cdist(ft, ft)

        # normaliza pela média
        d_s = d_s / (d_s.mean() + 1e-8)  
        d_t = d_t / (d_t.mean() + 1e-8)

        # huber e robusto a outliers
        return F.huber_loss(d_s, d_t)

    def _angulo(self, fs, ft):
        # Compara similaridade coseno do par-a-par (geometria angular).
        
        # projeta na esfera unitaria
        fs_norm = F.normalize(fs, dim=1)  
        ft_norm = F.normalize(ft, dim=1)

        # similaridade coseno: [B, B]
        sim_s = fs_norm @ fs_norm.T
        sim_t = ft_norm @ ft_norm.T
        return F.mse_loss(sim_s, sim_t)


class PerdaKDCosine(nn.Module):
    """
    Destilação via Cosine Embedding Loss + Cross-Entropy.

    Vantagem sobre MSE: Alinha a direção do vetor de features ignorando a magnitude.
    
    Isso é útil quando:
      - Teacher usa LayerNorm antes do classificador (ConvNeXt)
      - Student tem capacidade menor (não consegue reproduzir magnitudes exatas)
      - As escalas dos dois espaços de features são muito diferentes

    Fórmula:  L = (1 - cos(f_s, f_t)) + alpha_ce * L_CE

    Referência relacionada: Kim et al., CVPR 2018 (Paraphrasing Complex Network)
    """
    def __init__(self, alpha_ce: float = 1.0):
        super().__init__()
        self.alpha_ce = alpha_ce

    def forward(self, pred_student, feat_teacher, logits, labels):
        target = pred_student.new_ones(pred_student.size(0))  # maximizar similaridade
        l_cos = F.cosine_embedding_loss(pred_student, feat_teacher, target)
        l_ce  = F.cross_entropy(logits, labels)
        return l_cos + self.alpha_ce * l_ce, l_cos.item(), l_ce.item()


class PerdaKDCKA(nn.Module):
    """
    Centered Kernel Alignment (CKA) como perda de destilação.

    CKA mede a similaridade entre dois espaços de representação de forma
    invariante às transformações ortogonais e isotrópicas de escala.
    Isso o torna mais adequado que o MSE quando o teacher e o student têm
    espaços de features com estruturas geométricas muito diferentes.

    Fórmula linear (sem kernel):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    L_CKA = 1 - CKA(F_student, F_teacher)  (minimizar dissimilaridade)

    Referência: Kornblith et al., ICML 2019
                "Similarity of Neural Network Representations Revisited"
    """
    def __init__(self, alpha_ce: float = 1.0):
        super().__init__()
        self.alpha_ce = alpha_ce

    def _cka_linear(self, X, Y):
        # CKA linear centrado entre X e Y (ambos [B, D]).
        # centraliza as features no batch
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        num   = (Y.T @ X).norm(p='fro').pow(2)
        denom = (X.T @ X).norm(p='fro') * (Y.T @ Y).norm(p='fro') + 1e-8
        return num / denom

    def forward(self, pred_student, feat_teacher, logits, labels):
        l_cka = 1.0 - self._cka_linear(pred_student, feat_teacher)
        l_ce  = F.cross_entropy(logits, labels)
        return l_cka + self.alpha_ce * l_ce, l_cka.item(), l_ce.item()
