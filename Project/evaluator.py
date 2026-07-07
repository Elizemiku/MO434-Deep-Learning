# evaluator.py
# Avaliacao final do student e metricas de eficiencia

import numpy as np
import torch


class Evaluator:
    """
    Avaliação final do student e calculo de metricas de eficiência.
    """

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def avaliar_fase3(self, student, teacher, loader_test, modo_target='post_gap'):
        """
        Usa classificador do teacher para avaliar student.

        O classificador nunca viu as features do student durante o treinamento.
        Se o student aprendeu corretamente com as representações, o classificador 
        do teacher deve conseguir classificar as predicões do student com alta 
        acurácia (transferência de conhecimento).

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

            # top-5 (mais informativo para os datasets que tem muitas classes)
            if logits.shape[1] >= 5:
                top5 = logits.topk(5, dim=1).indices
                top5_accs.append((top5 == labels.unsqueeze(1)).any(1).float().mean().item())

        return np.mean(accs), (np.mean(top5_accs) if top5_accs else None)

    def contar_eficiencia(self, model, input_size=(1, 3, 224, 224)):
        """
        Retorna (gflops, n_parametros_M) do modelo.
        Precisa instalar o fvcore: pip install fvcore
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
        Imprime a tabela de eficiência: GFLOPs e parâmetros do teacher vs students.
        """
        print("Eficiência Computacional\n")
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
