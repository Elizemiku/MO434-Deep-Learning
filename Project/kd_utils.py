# kd_utils.py
# Classes auxiliares do projeto MO434 - Knowledge Distillation
#
#   set_seed  - fixa semente de aleatoriedade para reproducibilidade
#   Timer     - mede e registra o tempo de execucao de blocos de codigo

import random
import time
import functools

import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Fixa a semente de aleatoriedade em todos os modulos relevantes para garantir
    reproducibilidade: mesmos pesos iniciais, mesma ordem de batches e mesmos
    resultados entre execucoes diferentes.

    Modulos fixados:
      - random   (Python stdlib)
      - numpy    (operacoes de array)
      - torch    (CPU e CUDA)

    Parametros:
      seed: valor da semente (padrao=42). Deve ser o mesmo em todos os experimentos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # desativa otimizacoes nao-deterministicas da cuDNN que podem introduzir variacao
    # custo: leve reducao de velocidade na GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"semente fixada: {seed} (random, numpy, torch, cuda)")


# =============================================================================
# TIMER - MEDICAO DE TEMPO DE EXECUCAO
# =============================================================================

class Timer:
    """
    Mede o tempo de execucao de qualquer bloco de codigo no notebook.

    Uso como context manager (recomendado para celulas do notebook):

        with Timer("carregamento dos datasets"):
            dataset_manager.load_all()

        with Timer("fase 1 - resnet50"):
            historico = trainer.treinar_fase1(...)

    Uso como decorador:

        @Timer("minha funcao")
        def minha_funcao(): ...

    O tempo e impresso automaticamente ao fim do bloco no formato:
        [timer] carregamento dos datasets  ->  3.42 s
        [timer] fase 1 - resnet50          ->  2 min 14.8 s

    O historico de todas as medicoes fica em Timer.historico (dict global),
    permitindo comparar tempos ao final dos experimentos.
    """

    # historico global: acumula todos os tempos medidos na sessao
    historico: dict = {}

    def __init__(self, descricao: str = ""):
        self.descricao = descricao
        self._inicio   = None

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._inicio
        self._registrar(self.descricao, elapsed)
        return False  # nao suprime excecoes

    # ── decorador ─────────────────────────────────────────────────────────────

    def __call__(self, func):
        """Permite usar Timer como @Timer('nome') sobre uma funcao."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(self.descricao or func.__name__):
                return func(*args, **kwargs)
        return wrapper

    # ── metodos utilitarios ───────────────────────────────────────────────────

    @staticmethod
    def _registrar(descricao: str, elapsed: float):
        """Formata e imprime o tempo; salva no historico global."""
        if elapsed < 60:
            tempo_str = f"{elapsed:.2f} s"
        else:
            minutos = int(elapsed // 60)
            segundos = elapsed % 60
            tempo_str = f"{minutos} min {segundos:.1f} s"

        label = descricao if descricao else "bloco"
        print(f"[timer] {label:<45s} -> {tempo_str}")

        # acumula no historico global (sobrescreve se rodado mais de uma vez)
        Timer.historico[label] = elapsed

    @staticmethod
    def resumo():
        """
        Imprime tabela com todos os tempos medidos na sessao.
        Util para comparar quanto tempo cada fase/experimento consumiu.
        """
        if not Timer.historico:
            print("nenhuma medicao registrada ainda.")
            return
        print("\n" + "=" * 58)
        print(" Resumo de Tempos da Sessao")
        print("=" * 58)
        total = 0.0
        for label, elapsed in Timer.historico.items():
            if elapsed < 60:
                tempo_str = f"{elapsed:.2f} s"
            else:
                m = int(elapsed // 60); s = elapsed % 60
                tempo_str = f"{m} min {s:.1f} s"
            print(f"  {label:<45s} {tempo_str:>10s}")
            total += elapsed
        print("-" * 58)
        if total < 60:
            print(f"  {'TOTAL':<45s} {total:.2f} s")
        else:
            m = int(total // 60); s = total % 60
            print(f"  {'TOTAL':<45s} {m} min {s:.1f} s")
        print("=" * 58)
