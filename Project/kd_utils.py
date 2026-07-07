# kd_utils.py
# Classes auxiliares do projeto MO434 - Knowledge Distillation
#
#   set_seed  - fixa uma semente de aleatoriedade para que tenha reproducibilidade
#   Timer     - mede e registra o tempo de execucao dos blocos de codigo

import random
import time
import functools

import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Fixa uma semente de aleatoriedade em todos os modulos relevantes para garantir
    a reproducibilidade: mesmos pesos iniciais, mesma ordem de batches e mesmos
    resultados entre execuções diferentes.

    Modulos fixados:
      - random   (Python stdlib)
      - numpy    (operações de array)
      - torch    (CPU e CUDA)

    Parametros:
      seed: valor da semente (padrao=42). Deve ser o mesmo em todos os experimentos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # desativa otimizacoes não-deterministicas do cuDNN que podem criar variações entre execuções
    # custo: leve reducao de velocidade na GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"semente fixada: {seed} (random, numpy, torch, cuda)")


# Timer para medir o tempo de execução

class Timer:
    """
    Mede o tempo de execução de qualquer bloco de código no notebook.

    Uso como context manager:

        with Timer("carregamento dos datasets"):
            dataset_manager.load_all()

        with Timer("fase 1 - resnet50"):
            historico = trainer.treinar_fase1(...)

    Uso como decorador:

        @Timer("minha funcao")
        def minha_funcao(): ...

    O tempo é impresso automaticamente no fim do bloco neste formato:
        [timer] carregamento dos datasets  ->  3.42 s
        [timer] fase 1 - resnet50          ->  2 min 14.8 s

    O histórico de todas as medições fica em Timer.historico (dict global),
    permitindo comparar os tempos no final dos experimentos.
    """

    # histórico global: acumula todos os tempos medidos na sessão
    historico: dict = {}

    def __init__(self, descricao: str = ""):
        self.descricao = descricao
        self._inicio   = None

    # context manager

    def __enter__(self):
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._inicio
        self._registrar(self.descricao, elapsed)
        return False  # Para manter as exceções

    # decorador

    def __call__(self, func):
        # wrapper que permite usar o Timer como @Timer('nome').
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(self.descricao or func.__name__):
                return func(*args, **kwargs)
        return wrapper

    # métodos utilitários

    @staticmethod
    def _registrar(descricao: str, elapsed: float):
        # Formata e imprime o tempo
        if elapsed < 60:
            tempo_str = f"{elapsed:.2f} s"
        else:
            minutos = int(elapsed // 60)
            segundos = elapsed % 60
            tempo_str = f"{minutos} min {segundos:.1f} s"

        label = descricao if descricao else "bloco"
        print(f"[timer] {label:<45s} -> {tempo_str}")

        # Salva no histórico global (sobrescreve se rodar mais de uma vez)
        Timer.historico[label] = elapsed

    @staticmethod
    def resumo():
        """
        Imprime a tabela com todos os tempos medidos na sessão.
        Útil para comparar quanto tempo cada fase/experimento 
        demorou para executar.
        """
        if not Timer.historico:
            print("Sem nenhuma medição no momento.")
            return
        print("\n" + "=" * 58)
        print(" Resumo de tempos dessa sessão")
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
