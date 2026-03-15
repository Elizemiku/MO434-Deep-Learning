# Notas de Estudo — MO434 Deep Learning

## Dúvidas sobre o notebook Elizabeth-simple-classification

---

## 1) Por que a função de degrau (Sigmoid) se sairia melhor nesse caso?

As 3 classes são separadas por **duas linhas diagonais paralelas**. Os 4 neurônios da camada escondida calculam:

$$\text{neurônio}_i = \langle w_i, x \rangle + b_i$$

O que eles estão fazendo é: **"o ponto está acima ou abaixo da fronteira?"** — isso é uma pergunta binária (sim/não). A Sigmoid responde exatamente isso, mapeando a saída para [0, 1]:

- **Sigmoid** → saída limpa entre 0 e 1 → sinal claro para a camada de decisão
- **ReLU** → corta negativos em 0, mas valores positivos crescem sem limite → introduz diferença de magnitude desnecessária nesse problema

Para esse caso específico com fronteiras lineares, a Sigmoid cria **representações binárias mais limpas**, facilitando o trabalho da camada de decisão.

---

## 2) Código para plotar os hiperplanos dos neurônios

Cada neurônio `i` define uma reta: $w_{i,0} \cdot x_1 + w_{i,1} \cdot x_2 + b_i = 0$

Isolando $x_2$: $x_2 = \frac{-(w_{i,0} \cdot x_1 + b_i)}{w_{i,1}}$

```python
model.eval()
weights = model.classifier[0].weight.data.numpy()
biases  = model.classifier[0].bias.data.numpy()

fig, ax = plt.subplots(figsize=(7, 7))

# Plota os dados
for cluster, color, cls in zip(clusters, colors, classes):
    x1, x2 = cluster
    ax.scatter(x1, x2, alpha=0.8, c=color, edgecolors='none', s=50, label=cls)

# Plota o hiperplano de cada neurônio da camada escondida
x1_range = np.linspace(0.0, 1.5, 200)
neuron_colors = ['purple', 'orange', 'brown', 'pink']
for i in range(len(biases)):
    w1, w2 = weights[i]
    b = biases[i]
    if abs(w2) > 1e-6:
        x2_line = -(w1 * x1_range + b) / w2
        ax.plot(x1_range, x2_line, color=neuron_colors[i], label=f'Neurônio {i+1}')

ax.set_xlim(0.2, 1.3)
ax.set_ylim(0.2, 1.3)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
plt.title('Hiperplanos aprendidos pela camada escondida')
plt.show()
```

Esse código mostra onde cada neurônio "traça o corte" no espaço de entrada — é possível ver se a rede aprendeu fronteiras próximas das diagonais ideais.

---

## 3) Por que o fine-tuning piorou a acurácia?

A inicialização manual coloca os hiperplanos **exatamente** nas posições ótimas geométricas (calculadas com `b1` e `b2` que correspondem às distâncias reais entre as classes).

Quando `requires_grad = True`, o backpropagation tenta ajustar esses pesos "perfeitos" para minimizar ainda mais o loss do **treinamento**, e acaba:

- **Deslocando** ligeiramente os hiperplanos da posição ótima, gerando erro no **teste**
- Esse é o fenômeno clássico de **overfitting** — a rede superajusta ao treino e perde generalização

Em outras palavras: a geometria humana estava certa, e o otimizador acabou "estragando" algo que já era bom.

---

## 4) O que `requires_grad` controla no congelamento de camada?

```python
for param in model.classifier[0].parameters():
    param.requires_grad = False
```

| Valor | O que acontece |
|-------|----------------|
| `False` | Os pesos da camada 0 ficam **congelados** nos valores manuais. O SGD só atualiza a camada de decisão (camada 2). A rede **respeita sua geometria**. |
| `True` | Todos os pesos são treináveis. O backpropagation pode **modificar** os pesos da camada 0 a partir da sua inicialização manual. A rede faz do **jeito dela**. |

Ao congelar (`False`), a camada escondida mantém as fronteiras geométricas perfeitas e só a camada de decisão aprende **como combinar** esses sinais para classificar.

---

---

---

## Fundamentos de Redes Neurais

### O que é um MLP (Multi-Layer Perceptron)
- Cada neurônio faz $y = f(w \cdot x + b)$ — produto interno dos pesos com a entrada, mais bias, passado por uma função de ativação
- A rede tem 2 camadas lineares: **camada escondida** (2→4 neurônios) e **camada de decisão** (4→3 neurônios)

### Funções de ativação
- Por que precisamos delas: sem elas, a rede seria só uma transformação linear
- **ReLU**, **Sigmoid**, **LeakyReLU** — o que cada uma faz e quando usar

### Hiperplanos
- Cada neurônio define uma fronteira no espaço de entrada: $w_1 x_1 + w_2 x_2 + b = 0$
- O peso $w$ define a **orientação** da reta, o bias $b$ define o **deslocamento** da origem
- Isso é central para entender o notebook — os pesos manuais $(v, v)$ e $(-v, -v)$ são vetores diagonais a 45°

---

## Treinamento

### Função de perda (CrossEntropyLoss)
- Para classificação com múltiplas classes, ela mede o quão "errada" está a predição

### Backpropagation e SGD
- O gradiente da loss é propagado para trás → cada peso é ajustado na direção que reduz o erro
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` — esse ciclo é o coração do treino

### Xavier Initialization
- Inicializar pesos muito grandes ou muito pequenos causa gradientes explodindo ou sumindo
- Xavier escolhe valores proporcionais ao tamanho das camadas para estabilizar o início do treino

---

## Conceitos que o notebook testa especificamente

### Inicialização manual de pesos (a célula comentada)
- Você calcula geometricamente onde deveriam estar as fronteiras ideais e já começa a rede na solução correta
- Requer entender que $b_1 = \sqrt{0.85^2 + 0.85^2}$ é a distância do ponto $(0.85, 0.85)$ à origem

### Freezing de camadas (`requires_grad = False`)
- Separar o que deve aprender do que deve ficar fixo
- Base do conceito de **transfer learning** e fine-tuning

### Visualização dos hiperplanos
- Extrair pesos da rede e interpretar geometricamente onde cada neurônio colocou sua fronteira

---

## O que é necessário saber para compreender os fundamentos de Redes Neurais

---

## Matemática

### Álgebra Linear (o mais importante)
- Vetores, matrizes e multiplicação matricial — o `nn.Linear` nada mais é que $y = Wx + b$
- Produto interno $\langle w, x \rangle$ — é exatamente o que cada neurônio calcula
- Transposição, norma de vetores

### Cálculo
- Derivada de funções simples — necessária para entender o gradiente
- Regra da cadeia — é literalmente o que o backpropagation faz: derivar a loss em relação a cada peso camada por camada
- Gradiente — direção de maior crescimento de uma função

### Probabilidade e Estatística
- Distribuições de probabilidade básicas
- O que é esperança, variância
- Softmax e por que CrossEntropyLoss é adequada para classificação

---

## Programação

### Python intermediário
- Classes e orientação a objetos — toda rede em PyTorch é uma classe que herda de `nn.Module`
- NumPy — manipulação de arrays, que é a base de tudo

### Noção de otimização
- O conceito de **minimizar uma função** — o treino é basicamente isso
- O que é gradiente descendente: "andar na direção contrária ao gradiente para ir ao mínimo"

---

## Intuição geométrica

Isso vale especialmente para o notebook `Elizabeth-simple-classification`:
- Ponto no espaço 2D como vetor $(x_1, x_2)$
- O que é uma reta/hiperplano separando regiões do espaço
- Por que $w_1 x_1 + w_2 x_2 + b = 0$ define uma reta

---

## Ordem de estudo sugerida

```
Álgebra Linear → Cálculo (gradiente) → Gradiente Descendente
       ↓
   Neurônio único (Perceptron)
       ↓
   MLP com backpropagation (onde você está agora)
```

---

## Resumo: conceitos do notebook e onde aparecem

| Conceito | Onde aparece no notebook |
|---|---|
| Produto interno $w \cdot x + b$ | Definição do `nn.Linear` |
| Função de ativação | `nn.ReLU()` — a linha que você está estudando |
| Hiperplano / geometria | Pesos $(v, v)$ e bias $b_1, b_2$ |
| CrossEntropyLoss | Perda para 3 classes |
| SGD + backpropagation | Loop de treino |
| `requires_grad` | Congelar camada escondida |
| Acurácia treino vs. teste | Célula de avaliação |
