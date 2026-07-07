# trainer.py
# Loops de treino das 3 fases do projeto KD

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from kd_utils import set_seed
from losses import PerdaKD, PerdaRKD
from models import PlainCNNEncoder, PreditorPostGAP, StudentModel


class Trainer:
    """
    Gerencia os loops de treino para as três fases do projeto KD.

    Fases:
      Fase 1: Treinar o CLASSIFICADOR do teacher (encoder congelado)
      Fase 2: Treinar ENCODER + PREDITOR do student (imitação de features)
      Fase 3: (avaliação, ver Evaluator)
    """

    def __init__(self, device, seed: int = 42):
        """
        Parâmetros:
          device: torch.device ('cuda' ou 'cpu')
          seed:   a semente de aleatoriedade para reproducibilidade (padrao=42).
                  Usar seed=None para que a semente não seja fixada.
        """
        self.device = device
        self.seed   = seed
        if seed is not None:
            set_seed(seed)

    # helpers internos

    def _treinar_batch(self, model, dados, optimizer):
        """
        Executa um passo de treino: forward -> loss -> backward -> atualizar pesos.
        Retorna (loss, acuracia) do batch.
        """
        model.train()
        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        # Zera gradientes acumulados
        optimizer.zero_grad()
        # Forward pass                 
        logits = model(imgs)                     
        # Cross-entropy loss
        loss   = F.cross_entropy(logits, labels)
        # Backpropagation
        loss.backward()
        # Atualiza os pesos
        optimizer.step()

        acc = (logits.argmax(1) == labels).float().mean().item()
        return loss.item(), acc

    @torch.no_grad()
    def _validar_batch(self, model, dados):
        """
        Avalia o modelo sem computar os gradientes (é mais rapido e consome menos memória).
        model.eval() desativa o dropout e usa estatisticas fixas do batchnorm.
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
        # Avalia o modelo em um DataLoader completo, retornando a acurácia média.
        accs = []
        for dados in loader:
            _, a = self._validar_batch(model, dados)
            accs.append(a)
        return np.mean(accs)

    # Fase 1

    def treinar_fase1(self, model, loader_train, loader_val, optimizer, scheduler,
                      n_epochs, descricao="",
                      patience=5, overfitting_threshold=0.15):
        """
        Loop de treino completo com early stopping e detecção de overfitting.

        Na Fase 1:
          1. O encoder está congelado (nenhum gradiente passa por ele)
          2. Apenas o classificador é atualizado com Cross-Entropy Loss

        Por que isso é extremamente necessário para o KD?
        O classificador treinado aqui é reutilizado na Fase 3 para avaliar o student. 
        Isso força o student a produzir features que estejam no mesmo espaço do teacher.

        Detecção de overfitting:
          - gap_acc = acc_tr - acc_vl: mede a distância entre treino e validação.
            Quando gap_acc > overfitting_threshold por varias épocas, o modelo
            memorizou o treino e não generaliza.
          - val_loss subindo enquanto train_loss cai: sinal clássico de overfitting.
          - early stopping: para quando val_acc não melhora por "patience",
            restaurando os melhores pesos automaticamente.

        Parâmetros:
          patience:               épocas sem melhora antes de parar (0 = desativado)
          overfitting_threshold:  gap acc_tr - acc_vl que dispara aviso (padrão 0.15)
        """
        historico    = defaultdict(list)
        melhor_acc   = 0.0
        melhor_pesos = None
        melhor_epoch = 1
        # contador de epocas sem melhora (early stopping)
        sem_melhora  = 0 

        for epoch in range(1, n_epochs + 1):
            # treino
            losses_tr, accs_tr = [], []
            for dados in tqdm(loader_train, desc=f"[{descricao}] epoca {epoch}/{n_epochs}", leave=False):
                l, a = self._treinar_batch(model, dados, optimizer)
                losses_tr.append(l); accs_tr.append(a)

            # validacao
            losses_vl, accs_vl = [], []
            for dados in loader_val:
                l, a = self._validar_batch(model, dados)
                losses_vl.append(l); accs_vl.append(a)

            loss_tr = np.mean(losses_tr);  acc_tr = np.mean(accs_tr)
            loss_vl = np.mean(losses_vl);  acc_vl = np.mean(accs_vl)
            # gap de generalização: positivo indica que o modelo vai melhor no treino do que na validação. 
            # Valores altos (maiores que overfitting_threshold) sinalizam overfitting
            gap_acc = acc_tr - acc_vl

            historico['loss_tr'].append(loss_tr);  historico['acc_tr'].append(acc_tr)
            historico['loss_vl'].append(loss_vl);  historico['acc_vl'].append(acc_vl)
            # rastreia gaps de generalização
            historico['gap_acc'].append(gap_acc)

            if scheduler:
                scheduler.step()

            # detecção de overfitting
            # sinal 1: gap de acurácia alto e persistente
            # sinal 2: val_loss subindo enquanto train_loss cai (divergência clássica)
            avisos = []
            if gap_acc > overfitting_threshold:
                avisos.append(f"gap={gap_acc:.3f}>{overfitting_threshold}")
            if (len(historico['loss_vl']) >= 3 and
                    historico['loss_vl'][-1] > historico['loss_vl'][-2] > historico['loss_vl'][-3] and
                    historico['loss_tr'][-1] < historico['loss_tr'][-2]):
                avisos.append("val_loss subindo 3x seguidas")

            # early stopping
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
                print(f" Early stopping na época {epoch} "
                      f"(sem melhora por {patience} épocas). Melhor época: {melhor_epoch}")
                break

        # restaura melhores pesos
        if melhor_pesos:
            model.load_state_dict(melhor_pesos)
        # lista para compatibilidade com defaultdict
        historico['melhor_epoch'] = [melhor_epoch]
        print(f" Melhor acc validação: {melhor_acc:.4f} (época {melhor_epoch})")
        return historico

    # fase 2: destilação

    def _treinar_batch_kd(self, student, teacher, dados, optimizer, perda_fn,
                          modo_target='post_gap'):
        """
        Cada passo de treino de destilação:
          1. Extrai as features do teacher (sem gradiente)
          2. O student prediz essas features
          3. Calcula a perda combinada MSE + CE
          4. Retropropaga e atualiza apenas os pesos do student

        Ponto crítico: NUNCA deixar os gradientes fluirem pelo encoder do teacher.
        torch.no_grad() serve para economizar memória e tempo de computação.
        """
        student.train()
        # teacher sempre em eval (batchnorm usa estatisticas fixas)
        teacher.eval()

        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        # extrai targets do teacher sem gradiente (economiza memória)
        with torch.no_grad():
            if modo_target == 'post_gap':
                feat_teacher = teacher.get_post_gap(imgs)   # [B, C_t]
            else:
                feat_teacher = teacher.get_pre_gap(imgs)    # [B, C_t, 7, 7]

        optimizer.zero_grad()
        pred_student = student(imgs)   # [B, C_t] ou [B, C_t, 7, 7]

        # passa a predição pelo classificador do teacher para obter logits
        # IMPORTANTE: não usar torch.no_grad() aqui, pois o gradiente da CE
        # precisa fluir até pred_student para guiar o espaço de features.
        # Apenas desabilitamos grad nos parâmetros do classificador (que não são otimizados).
        for p in teacher.classifier.parameters():
            p.requires_grad_(False)
        if modo_target == 'post_gap':
            logits = teacher.classifier(pred_student)
        else:
            pooled = teacher.gap(pred_student).flatten(1)
            logits = teacher.classifier(pooled)

        perda, l_mse, l_ce = perda_fn(pred_student, feat_teacher, logits, labels)
        perda.backward()
        # clipa gradientes para evitar explosão (importante para redes profundas)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        acc = (logits.argmax(1) == labels).float().mean().item()
        return perda.item(), l_mse, l_ce, acc

    @torch.no_grad()
    def validar_student(self, student, teacher, loader_val, modo_target='post_gap'):
        # avalia o student usando o classificador do teacher (preparação para fase 3).
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
        Avalia o student retornando (acc_vl, loss_vl) usando a perda de destilação.
        Necessário para detectar divergências de val_loss da Fase 2 (sinal de overfitting).
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
        Loop completo de destilação com registro de métricas e detecção de overfitting.

        O teacher está completamente congelado. O student aprende a imitar as features do
        teacher através do gradiente da perda de destilação.

        Fluxo:
          Imagem -> [TEACHER ENCODER, congelado] -> features_teacher (alvo, sem grad)
          Imagem -> [STUDENT ENCODER, treina]    -> features_student
                 -> [PREDITOR, treina]           -> pred_features
                                                       |
                                            MSE(pred_features, features_teacher)
                                            + CE(classifier(pred_features), labels)

        Detecção de overfitting:
          - gap_acc = acc_tr - acc_vl: distância entre treino e validação.
          - val_loss subindo enquanto train_loss cai: divergência clássica.
          - early stopping por paciência: para quando val_acc não muda.

        Parâmetros:
          patience:               épocas sem melhora antes de parar (0 = desativado)
          overfitting_threshold:  gap acc_tr - acc_vl que dispara aviso de overfitting (padrão 0.15)
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

            # avalia val com perda para monitorar divergência de loss
            acc_vl, loss_vl = self._validar_student_kd(
                student, teacher, loader_val, perda_fn, modo_target)
            scheduler.step()

            acc_tr  = np.mean(accs_tr)
            loss_tr = np.mean(perdas)
            # gap de generalização: positivo indica melhor desempenho no treino do que na validação
            gap_acc = acc_tr - acc_vl

            historico['perda'].append(loss_tr)
            historico['loss_vl'].append(loss_vl)
            historico['mse'].append(np.mean(mselist))
            historico['acc_tr'].append(acc_tr)
            historico['acc_vl'].append(acc_vl)
            historico['gap_acc'].append(gap_acc)  # rastreia gap de generalização

            # detecção de overfitting
            avisos = []
            if gap_acc > overfitting_threshold:
                avisos.append(f"gap={gap_acc:.3f}>{overfitting_threshold}")
            if (len(historico['loss_vl']) >= 3 and
                    historico['loss_vl'][-1] > historico['loss_vl'][-2] > historico['loss_vl'][-3] and
                    historico['perda'][-1] < historico['perda'][-2]):
                avisos.append("val_loss subindo 3x seguidas")

            # early stopping
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
                print(f" Early stopping na época {epoch} "
                      f"(sem melhora por {patience} épocas). Melhor época: {melhor_epoch}")
                break

        if melhor_pesos:
            student.load_state_dict(melhor_pesos)
        historico['melhor_epoch'] = [melhor_epoch]
        print(f" Melhor acc validação: {melhor_acc:.4f} (época {melhor_epoch})")
        return historico, melhor_acc

    # ── Q5: comparacao mse vs rkd ─────────────────────────────────────────────

    def _treinar_batch_rkd(self, student, teacher, dados, optimizer, perda_rkd_fn,
                           alpha_ce=1.0):
        """
        Passo de treino com RKD:
        l_total = alpha_ce * CE(logits, labels) + RKD(feat_student, feat_teacher)

        Seguindo Park et al. (CVPR 2019): CE e a perda principal (ancora o espaço
        de features) e RKD e o regularizador relacional. alpha_ce=1.0 garante que
        os gradientes de CE tenham peso total na otimização do student.
        """
        student.train(); teacher.eval()
        imgs, labels = dados
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        with torch.no_grad():
            feat_teacher = teacher.get_post_gap(imgs)

        optimizer.zero_grad()
        pred_student = student(imgs)

        # rkd compara relações par-a-par no espaço de features
        l_rkd = perda_rkd_fn(pred_student, feat_teacher)

        # CE supervisionado: Congelamos o classificador do teacher para que o
        # backward não acumule gradientes inúteis nos seus pesos (eles não estão
        # no optimizer e nunca são atualizados). O gradiente da CE ainda flui
        # corretamente até pred_student através dos pesos congelados.
        for p in teacher.classifier.parameters():
            p.requires_grad_(False)
        logits = teacher.classifier(pred_student)
        l_ce  = F.cross_entropy(logits, labels)

        # formula RKD paper: CE primária + RKD regularizador
        perda = alpha_ce * l_ce + l_rkd
        perda.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        acc = (logits.argmax(1) == labels).float().mean().item()
        return perda.item(), l_rkd.item(), l_ce.item(), acc

    def comparar_mse_vs_rkd(self, teacher, datasets, teacher_nome, dataset_nome,
                             n_epochs=20, lr=1e-3, student_channels=(32, 64, 128, 256)):
        """
        Compara MSE baseline vs RKD com o mesmo encoder e teacher (Q5).

        MSE força o student a copiar os valores absolutos das features.
        RKD alinha as relações entre amostras, independente da escala.
        Útil quando o student tem capacidade menor (capacity mismatch).

        Retorna dict com accs_vl e melhor_acc para cada método.
        """
        resultados_comp = {}

        for nome_perda, modo in [('MSE_baseline', 'mse'), ('RKD', 'rkd')]:
            print(f"  treinando: {nome_perda}")

            enc    = PlainCNNEncoder(student_channels)
            pred   = PreditorPostGAP(enc.out_dim, teacher.feat_dim)
            stu    = StudentModel(enc, pred).to(self.device)
            optim_ = optim.AdamW(stu.parameters(), lr=lr, weight_decay=1e-4)
            sched_ = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=n_epochs)

            # garante que classifier do teacher esteja na GPU e sem grad
            # (o congelamento individual é feito em _treinar_batch_rkd, mas garantimos
            # aqui para o modo MSE também, pois _treinar_batch_kd usa no_grad)
            teacher.to(self.device)

            perda_fn_mse = PerdaKD(alpha=0.5)
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
