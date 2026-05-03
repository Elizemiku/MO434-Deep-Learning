# trainer.py
# Loops de treino das 3 fases do projeto KD - MO434
#
# Fases:
#   Fase 1: Treinar o CLASSIFICADOR do teacher (encoder congelado)
#   Fase 2: Treinar ENCODER + PREDITOR do student (imitacao de features)
#   Fase 3: (avaliacao, ver evaluator.py)
#
# Inclui:
#   - deteccao de overfitting (gap de acuracia, divergencia de val_loss)
#   - early stopping com paciencia configuravel
#   - rastreamento completo de metricas por epoca

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from kd_helpers import set_seed
from losses import PerdaKD, PerdaRKD
from models import PlainCNNEncoder, PreditorPostGAP, StudentModel


class Trainer:
    """
    Gerencia loops de treino para as tres fases do projeto KD.

    Fases:
      Fase 1: Treinar o CLASSIFICADOR do teacher (encoder congelado)
      Fase 2: Treinar ENCODER + PREDITOR do student (imitacao de features)
      Fase 3: (avaliacao, ver Evaluator)
    """

    def __init__(self, device, seed: int = 42):
        """
        Parametros:
          device: torch.device ('cuda' ou 'cpu')
          seed:   semente de aleatoriedade para reproducibilidade (padrao=42).
                  Use seed=None para desativar a fixacao de semente.
        """
        self.device = device
        self.seed   = seed
        if seed is not None:
            set_seed(seed)

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
