# Contexto do Projeto MO434 — Knowledge Distillation

> Arquivo de contexto para continuar o trabalho neste repositório.
> Gerado em 2026-07-02 após sessão de desenvolvimento completa.

---

## 1. O que é este projeto

Projeto final da disciplina **MO434 / MC934 — Deep Learning** (UNICAMP, Prof. Alexandre Xavier Falcão).

O objetivo é implementar e avaliar esquemas de **Knowledge Distillation (KD)** que transferem o conhecimento de redes convolucionais pré-treinadas (teachers: VGG-16, ResNet-50, ConvNeXt-Small) para um encoder leve treinado do zero (student), usando os datasets **Flowers-102** e **Oxford-IIIT-Pet**.

### Pipeline de 3 fases

```
Fase 1 → Treinar classificador do teacher (encoder congelado, 15 épocas)
Fase 2 → Treinar encoder + preditor MLP do student (imitação de features post-GAP)
Fase 3 → Avaliar student com o classificador da Fase 1 (sem re-treino)
```

### 5 questões de pesquisa respondidas

| Q | Pergunta | Melhor resposta |
|---|----------|-----------------|
| Q1 | Qual teacher transfere melhor? | ResNet-50 (Flowers-102), VGG-16 (Pets) |
| Q2 | pre-GAP vs post-GAP? | post-GAP (≈2,5× superior, preditor mais leve) |
| Q3 | Melhor arquitetura de encoder? | Depthwise (ResNet-50), PlainCNN (VGG-16, ConvNeXt) |
| Q4 | Melhor α em MSE+CE? | α=0,5 |
| Q5 | MSE vs RKD? | RKD amplamente superior (CE estava bloqueado no baseline) |

---

## 2. Estrutura do repositório

```
Project/
├── models/
│   ├── blocks.py         # conv_block (com dw=True para depthwise), ResBlock
│   ├── encoders.py       # PlainCNNEncoder, DepthwiseCNNEncoder, MiniResNetEncoder
│   ├── predictors.py     # PreditorPostGAP(dim_s, dim_t, activation='relu'|'silu')
│   │                     # PreditorPreGAP, StudentModel(encoder, preditor)
│   └── teacher.py        # TeacherWrapper(nome, n_classes)
│                         #   .freeze_encoder(), .get_post_gap(x), .get_pre_gap(x)
│                         #   .feat_dim, .classifier
├── losses.py             # PerdaKD(alpha), PerdaRKD, PerdaKDCosine, PerdaKDCKA
├── trainer.py            # Trainer(device, seed) — loops Fase1/2/3
├── evaluator.py          # Evaluator — top-1 e top-5
├── load_datasets.py      # DatasetManager → DATASETS['flowers102'|'pets']
├── transforms.py         # ImageTransforms (ImageNet mean/std)
├── visualization.py      # plotar_curvas, plotar_comparacao_mse_rkd, etc.
├── kd_run.py             # re-exporta tudo (único ponto de import dos notebooks)
├── kd_utils.py           # utilitários (Timer, set_seed, etc.)
│
├── experimento_final_KD_test.ipynb   ← NOTEBOOK PRINCIPAL (seções 1–12)
├── guia_knowledge_distillation_t3.ipynb  ← notebook de guia/ablação
│
├── checkpoints/          # modelos .pth, sumários .json, resultados .pkl/.csv
│   ├── classifier_{teacher}_{dataset}.pth          # Fase 1
│   ├── best_student_{teacher}_{dataset}.pth        # Fase 2 (ReLU)
│   ├── best_student_resnet50_{dataset}_silu.pth    # Fase 2 (SiLU, Seção 11)
│   ├── results_q1.csv, results_q3.csv
│   ├── results_s10_ablacao.pkl
│   └── results_silu_complement.pkl
│
└── Latex/
    ├── main.tex          ← relatório LaTeX (compilar com pdflatex)
    ├── referencias.bib   ← bibliografia (15+ entradas)
    └── Figuras/          ← todas as figuras .pdf e .png
        ├── fig_q1.{pdf,png}              # Q1 teacher comparison
        ├── fig_q3.{pdf,png}              # Q3 encoder comparison
        ├── fig_q5.{pdf,png}              # Q5 MSE vs RKD bar chart
        ├── fig_s10_ablacao.{pdf,png}     # Seção 10 ablação (barras + curvas)
        ├── fig_s11_silu.{pdf,png}        # Seção 11 ReLU vs SiLU convergência
        ├── fig_convergencia.{pdf,png}    # curvas best student por teacher
        ├── fig_predicoes_flowers.{pdf,png}  # predições qualitativas Flowers-102
        ├── fig_predicoes_pets.{pdf,png}     # predições qualitativas Pets
        ├── fig_guia_analise_q1q3q4.png   # análise visual Q1/Q3/Q4 (guia nb)
        ├── fig_guia_q5_rkd.png           # convergência MSE vs RKD (guia nb)
        ├── fig_guia_fase1_{teacher}.png  # curvas treino Fase 1 (guia nb)
        └── fig_guia_amostras.png         # amostras dos 2 datasets (guia nb)
```

---

## 3. Ambiente Python

- **Conda env**: `mo434`
- **Python**: 3.12.8
- **Executável**: `/home/beth/miniconda3/envs/mo434/bin/python`
- **Device**: CUDA (GPU disponível)
- **Pacotes principais**: PyTorch, torchvision, pandas, matplotlib, seaborn, scikit-learn

Ativar: `conda activate mo434`

---

## 4. Resultados obtidos (estado atual)

### Fase 1 — Acurácia teacher (15 épocas, encoder congelado)

| Teacher | Flowers-102 | Oxford-IIIT-Pet |
|---------|-------------|-----------------|
| VGG-16 | 77,0% | 89,3% |
| ResNet-50 | 84,1% | 90,4% |
| ConvNeXt-Small | 86,9% | 93,3% |

### Q1/Q3 — Best student por teacher (50 épocas, ReLU)

| Teacher | Encoder | Params | Flowers Top-1 | Flowers Top-5 | Pets Top-1 |
|---------|---------|--------|--------------|--------------|------------|
| ResNet-50 | Depthwise | 1,23M | **30,7%** | 59,1% | 24,0% |
| VGG-16 | PlainCNN | 0,78M | 27,7% | 55,6% | **37,7%** |
| ConvNeXt-Small | PlainCNN | 0,91M | 6,6% | 24,8% | 24,5% |

### Seção 10 — Ablação de estratégias avançadas (ResNet-50, Flowers, 30 épocas)

| Variante | Melhor Acc Val | Δ vs baseline |
|----------|---------------|---------------|
| MSE+CE (baseline corrigido) | 39,7% | — |
| RKD+CE | 39,9% | +0,2 p.p. |
| **MSE+CE + SiLU** | **42,9%** | **+3,2 p.p.** |
| Cosine+CE | 39,5% | −0,2 p.p. |
| MSE+CE + SE-Atenção | 39,5% | −0,2 p.p. |
| CKA+CE | 38,3% | −1,4 p.p. |

### Seção 11 — SiLU definitivo (ResNet-50/Depthwise, 50 épocas)

| Dataset | Ativação | Acc Val | Top-1 Teste | Top-5 Teste | Δ Top-1 |
|---------|----------|---------|------------|------------|---------|
| Flowers-102 | ReLU | 37,5% | 30,7% | 59,1% | — |
| Flowers-102 | **SiLU** | **50,3%** | **44,1%** | **71,6%** | +13,4 p.p. |
| Oxford-IIIT-Pet | ReLU | 31,7% | 24,0% | 59,5% | — |
| Oxford-IIIT-Pet | **SiLU** | **40,8%** | **33,9%** | **70,7%** | +9,9 p.p. |

---

## 5. Detalhes técnicos importantes

### PlainCNNEncoder — atributo correto
```python
# CORRETO: usa self.features (nn.Sequential)
enc.features  # ← usa isso

# ERRADO (causa AttributeError):
enc.blocks    # ← NÃO existe
```

### PreditorPostGAP — suporta ativação SiLU
```python
from models.predictors import PreditorPostGAP
pred = PreditorPostGAP(dim_student=256, dim_teacher=2048, dim_hidden=512, activation='silu')
# activation: 'relu' (default) | 'gelu' | 'silu'
```

### Losses disponíveis (losses.py + kd_run.py)
```python
from kd_run import PerdaKD, PerdaRKD, PerdaKDCosine, PerdaKDCKA
from models.blocks import SEBlock  # SEBlock(channels, reduction=16)
```

### Problema crítico corrigido no trainer.py (RKD)
O `_treinar_batch_rkd` originalmente calculava logits dentro de `torch.no_grad()`,
bloqueando o gradiente CE. **Solução aplicada**: remover o `no_grad` dos logits
e desabilitar gradiente só no classificador com `.requires_grad_(False)`.

### Variáveis disponíveis no kernel do notebook principal
Após executar todas as células do notebook, o kernel tem:
- `DATASETS`, `TEACHERS`, `DATASETS_NOMES`
- `resultados_fase1[teacher][dataset]` — `{teacher_obj, historico, acc_test}`
- `resultados_final[teacher][dataset]` — best student ReLU
- `resultados_q5` — MSE vs RKD
- `resultados_s10` — ablação Seção 10 (6 variantes)
- `resultados_silu` — best student SiLU por dataset
- `df_rel`, `df_q3`, `df_eff` — DataFrames de resultados

---

## 6. Como atualizar o relatório LaTeX com novos resultados

### 6.1 Workflow geral

```
1. Treinar/executar experimento no notebook
2. Atualizar variáveis de resultados no kernel
3. Re-executar a célula de exportação de figuras (última célula do notebook)
4. Copiar figuras para Project/Latex/Figuras/ se necessário
5. Editar os valores nas tabelas do main.tex
6. Compilar: pdflatex main.tex (no diretório Project/Latex/)
```

### 6.2 Célula de exportação de figuras

A **última célula do notebook** (`experimento_final_KD_test.ipynb`, Seção 12)
exporta todas as figuras automaticamente para `Project/Latex/Figuras/`.

> **Atenção**: O `FIG_DIR` deve ser `'Latex/Figuras'` (relativo ao cwd `Project/`),
> NÃO `'../Latex/Figuras'` (que salvaria um nível acima na estrutura errada).

Para re-exportar após novos experimentos:
```python
# Basta re-executar a última célula do notebook com o kernel ativo
# Ela usa as variáveis do kernel: resultados_final, resultados_silu, resultados_s10, etc.
```

### 6.3 Atualizar uma tabela de resultados no main.tex

Localizar pelo label e editar os valores:

```latex
% Tabela Q1 (resultados finais por teacher)
\label{tab:q1}   % linha ~535

% Tabela ablação Seção 10
\label{tab:s10}  % linha ~727

% Tabela ReLU vs SiLU
\label{tab:s11}  % linha ~767
```

### 6.4 Adicionar uma nova figura

1. Gerar e salvar a figura em `Project/Latex/Figuras/nome_fig.pdf` (e `.png`)
2. No `main.tex`, inserir:

```latex
\begin{figure}[htb]
\centering
\includegraphics[width=\linewidth]{nome_fig}
\caption{Descrição da figura.}
\label{fig:nome_fig}
\end{figure}
```

O `\graphicspath{ {./Figuras/} }` já está configurado no preâmbulo.

### 6.5 Adicionar uma nova seção de resultados

Exemplo de como a Seção 10/11 foi adicionada (padrão a seguir):

```latex
\subsection{Nome da Nova Seção}
\label{sec:sXX}

Descrição do experimento...

\begin{table}[htb]
\centering
\caption{Título da tabela.}
\label{tab:sXX}
\begin{tabular}{@{}lcc@{}}
\toprule
Coluna 1 & Coluna 2 & Coluna 3 \\ \midrule
Linha 1  & val1     & val2     \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htb]
\centering
\includegraphics[width=\linewidth]{fig_sXX}
\caption{Legenda.}
\label{fig:sXX}
\end{figure}
```

### 6.6 Adicionar uma nova entrada bibliográfica

Editar `Project/Latex/referencias.bib` com uma nova entrada BibTeX e
citar no texto com `\cite{chave}` ou `\textcite{chave}`.

---

## 7. Estrutura atual do main.tex

```
§1  Fundamentação Teórica
    §1.1 Destilação de Conhecimento (Hinton, FitNets, RKD, survey)
    §1.2 Arquiteturas Teacher (VGG-16, ResNet-50, ConvNeXt-Small)
    §1.3 Conjuntos de Dados (Flowers-102, Oxford-IIIT-Pet)

§2  Implementação e Metodologia
    §2.1 Pipeline de Três Fases (TikZ diagram)
    §2.2 Arquitetura do Student
    §2.3 Funções de Perda (PerdaKD, PerdaRKD)
    §2.4 Estrutura do Repositório (tabela + TikZ diagram de dependências)
    §2.5 Dificuldades Encontradas e Soluções

§3  Resultados e Discussão
    §3.1 Q1 — Qual Teacher Transfere Melhor?     [tab:fase1, tab:q1, fig:q1]
    §3.2 Q2 — pre-GAP vs post-GAP                [tab:q2]
    §3.3 Q3 — Melhor Encoder Student             [tab:q3, fig:q3, fig:guia_analise]
    §3.4 Q4 — Ablação α                          [tabela inline]
    §3.5 Q5 — MSE vs RKD                         [tab:q5, fig:q5, fig:guia_q5]
    §3.6 Ablação Estratégias Avançadas + SiLU    [tab:s10, fig:s10, tab:s11, fig:s11]
    §3.7 Eficiência Computacional                [tab:eff]
    §3.8 Análise Qualitativa                     [fig:pred_flowers, fig:pred_pets, fig:convergencia]

§4  Conclusão (6 itens: Q1–Q5 + SiLU)
§5  Sugestões para Aprimoramento (6 itens)
```

---

## 8. Compilar o relatório

```bash
cd /home/beth/projetos/MO434-Deep-Learning/Project/Latex

# Se texlive não estiver instalado:
sudo apt install texlive-full biber

# Compilar (2 passagens para referências cruzadas):
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

---

## 9. Próximos passos possíveis

- [ ] Compilar o PDF final e revisar a diagramação
- [ ] Executar SiLU também para VGG-16 e ConvNeXt-Small (atualizar tab:q1)
- [ ] Adicionar t-SNE figure (código existe no notebook, Seção 9.2)
- [ ] Corrigir o baseline MSE em Q5 (remover `no_grad` dos logits) para comparação justa
- [ ] Aplicar augmentation mais agressiva (RandAugment) — Flowers-102 tem apenas 1020 imgs de treino
- [ ] Tentar SiLU + RKD combinados
