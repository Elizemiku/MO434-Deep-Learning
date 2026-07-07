# visualization.py
# Graficos e visualizacoes para analise dos experimentos

import matplotlib.pyplot as plt
import numpy as np


def plotar_curvas(historico, titulo="", overfitting_threshold=0.15):
    """
    Plota as curvas de loss, acuracia e gap de generalização.

    O terceiro gráfico (gap) facilita a leitura do overfitting:
      - gap = acc_treino - acc_validacao
      - valores acima da linha vermelha tracejada (threshold) indicam overfitting
      - linha vertical cinza marca a melhor época (early stopping ou melhor val_acc)
    """
    tem_gap = 'gap_acc' in historico and len(historico['gap_acc']) > 0
    n_plots = 3 if tem_gap else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 1, 4))
    if n_plots == 2:
        axes = list(axes)

    # marca a melhor época, se disponível
    melhor_epoch = None
    if 'melhor_epoch' in historico and historico['melhor_epoch']:
        melhor_epoch = historico['melhor_epoch'][0] - 1  # índice 0-based

    def _marcar_melhor(ax):
        if melhor_epoch is not None:
            ax.axvline(melhor_epoch, color='gray', linestyle='--', linewidth=1,
                       alpha=0.7, label=f'melhor (ep.{melhor_epoch + 1})')
            # sombreia a região após a melhor época como zona de possível overfitting
            if melhor_epoch < len(historico['acc_vl']) - 1:
                ax.axvspan(melhor_epoch, len(historico['acc_vl']) - 1,
                           alpha=0.07, color='red', label='zona após melhor época')

    # gráfico loss
    ax = axes[0]
    epocas = range(len(historico['loss_tr']))
    ax.plot(epocas, historico['loss_tr'], label='treino')
    key_lv = 'loss_vl' if 'loss_vl' in historico else 'perda'
    if key_lv in historico and len(historico[key_lv]) == len(historico['loss_tr']):
        ax.plot(epocas, historico[key_lv], label='validacao')
    _marcar_melhor(ax)
    ax.set_title(f'loss - {titulo}'); ax.set_xlabel('epoca')
    ax.legend(); ax.grid(True, alpha=0.3)

    # gráfico 2: acurácia
    ax = axes[1]
    epocas = range(len(historico['acc_tr']))
    ax.plot(epocas, historico['acc_tr'], label='treino')
    ax.plot(epocas, historico['acc_vl'], label='validação')
    _marcar_melhor(ax)
    ax.set_title(f'acurácia - {titulo}'); ax.set_xlabel('época')
    ax.legend(); ax.grid(True, alpha=0.3)

    # gráfico 3: gap de generalização (acc_tr - acc_vl)
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
        ax.set_title(f'gap de generalização - {titulo}')
        ax.set_xlabel('época'); ax.set_ylabel('acc_treino - acc_val')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()


def plotar_comparacao_mse_rkd(resultados_comp, teacher_nome, dataset_nome):
    # Plota a comparação de curvas de validação entre MSE baseline e RKD.
    plt.figure(figsize=(8, 4))
    for nome, res in resultados_comp.items():
        plt.plot(res['accs_vl'], label=f"{nome} (melhor={res['melhor_acc']:.3f})")
    plt.xlabel('época'); plt.ylabel('acurácia validação')
    plt.title(f'MSE vs RKD | teacher={teacher_nome}, dataset={dataset_nome}')
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()


def plotar_graficos_analise(df_resultados):
    """
    Gráficos de análise final respondendo Q1, Q3 e Q4:
      - Q1: comparação de teachers
      - Q4: ablação do peso alpha
      - Q3: trade-off acurácia vs parâmetros
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Q1: comparação de teachers
    ax = axes[0]
    dados_q1 = df_resultados[
        (df_resultados['encoder'] == 'PlainCNNEncoder') &
        (df_resultados['target']  == 'post_gap') &
        (df_resultados['alpha']   == 0.7)
    ].sort_values('acc_test', ascending=True)
    ax.barh(dados_q1['teacher'], dados_q1['acc_test'], color='steelblue', alpha=0.8)
    ax.set_title('Q1 - Qual teacher transfere melhor?', fontsize=11)
    ax.set_xlabel('acurácia no teste'); ax.grid(True, alpha=0.3, axis='x')

    # Q4: Exploração de alpha
    ax = axes[1]
    dados_q4 = df_resultados[
        (df_resultados['teacher'] == 'resnet50') &
        (df_resultados['encoder'] == 'PlainCNNEncoder') &
        (df_resultados['target']  == 'post_gap')
    ].sort_values('alpha')
    ax.plot(dados_q4['alpha'], dados_q4['acc_test'], 'o-', color='darkorange', linewidth=2)
    ax.set_title('Q4 - Ablação do peso alpha', fontsize=11)
    ax.set_xlabel('alpha (peso do MSE)'); ax.set_ylabel('acurácia no teste')
    ax.grid(True, alpha=0.3)

    # Q3: acurácia vs parâmetros
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
    ax.set_title('Q3 - Trade-off: acurácia vs parâmetros', fontsize=11)
    ax.set_xlabel('parâmetros do student (M)'); ax.set_ylabel('acurácia no teste')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Análise dos Experimentos - Knowledge Distillation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
