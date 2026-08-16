# DEFINIÇÕES TÉCNICAS DE INTELIGÊNCIA ARTIFICIAL

## INTRODUÇÃO

Este documento fornece uma compreensão profunda dos conceitos técnicos que fundamentam a Inteligência Artificial moderna. Exploraremos como os algoritmos funcionam "por debaixo dos panos", desde conceitos matemáticos básicos até arquiteturas complexas que potencializam sistemas de IA contemporâneos.

---

## SEÇÃO 1: FUNDAMENTOS MATEMÁTICOS

### 1.1 Redes Neurais Artificiais

#### **O que é um Neurônio Artificial?**

Um neurônio artificial é uma unidade computacional que imita o comportamento de um neurônio biológico. Funciona em três etapas:

**Estrutura Básica:**

```
Entradas (x₁, x₂, ..., xₙ)
    ↓
Pesos (w₁, w₂, ..., wₙ)  [multiplicação]
    ↓
Soma: z = Σ(wᵢ * xᵢ) + b  [bias b]
    ↓
Função de Ativação: a = f(z)
    ↓
Saída (a)
```

**Componentes Detalhados:**

1. **Entradas (x)**: Dados numéricos que fluem para o neurônio
2. **Pesos (w)**: Parâmetros aprendíveis que amplificam ou reduzem a importância de cada entrada
3. **Bias (b)**: Deslocamento que permite ao neurônio ativar mesmo com todas entradas zero
4. **Soma Ponderada**: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
5. **Função de Ativação**: Introduz não-linearidade, permitindo à rede aprender padrões complexos

#### **Funções de Ativação Comuns**

**Sigmoid:**
```
f(z) = 1 / (1 + e^(-z))
```
- Saída entre 0 e 1
- Historicamente popular, agora menos usada
- Problema: Vanishing Gradient (gradientes desaparecem em valores extremos)

**Tanh (Tangente Hiperbólica):**
```
f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
- Saída entre -1 e 1
- Melhor que sigmoid, ainda sofre com vanishing gradient

**ReLU (Rectified Linear Unit):**
```
f(z) = max(0, z)
```
- Muito popular em deep learning moderno
- Computacionalmente eficiente
- Problema: Dead ReLU (neurônios podem "morrer" com valor 0)
- Solução: Leaky ReLU: f(z) = max(0.01z, z)

**Softmax (para classificação multiclasse):**
```
f(zᵢ) = e^(zᵢ) / Σⱼ(e^(zⱼ))
```
- Converte saídas em probabilidades
- Soma a 1, interpretável como distribuição de probabilidade

#### **Como Redes Neurais Aprendem: Backpropagation**

Backpropagation é o algoritmo fundamental que permite redes neurais aprender:

**Processo em 4 Etapas:**

1. **Forward Pass (Passe para Frente)**
   - Entrada flui através da rede
   - Cada neurônio calcula sua saída
   - Resultado final comparado com resposta correta
   - Calcula-se erro/loss: L = (ŷ - y)²

2. **Cálculo do Erro**
   - Loss (Perda): Mede distância entre previsão (ŷ) e valor real (y)
   - Objetivo: Minimizar loss

3. **Backward Pass (Passe para Trás)**
   - Erro flui de volta pela rede
   - Para cada peso, calcula-se gradiente: ∂L/∂w
   - Usa regra da cadeia de cálculo diferencial

4. **Atualização de Pesos**
   - Pesos são ajustados na direção oposta ao gradiente
   - w_novo = w_antigo - α * ∂L/∂w
   - α é a "taxa de aprendizado" (learning rate)

**Intuição:**
- Se aumentar w aumenta erro, diminua w
- Se aumentar w diminui erro, aumente w
- A magnitude do ajuste é proporcional ao quão "errada" a previsão estava

#### **Exemplo Prático: Classificação Binária**

Suponha uma rede com:
- 2 entradas (x₁, x₂)
- 1 neurônio escondido
- 1 neurônio de saída

```
Input: [0.5, 0.3]
Pesos layer 1: w₁=0.1, w₂=0.2, b=0.1
z₁ = 0.1*0.5 + 0.2*0.3 + 0.1 = 0.17
a₁ = ReLU(0.17) = 0.17

Pesos layer 2: w₃=0.5, b=0.2
z₂ = 0.5*0.17 + 0.2 = 0.285
output = Sigmoid(0.285) ≈ 0.57

Target = 1.0 (classe positiva)
Loss = (0.57 - 1.0)² = 0.1849

Gradientes calculados via backprop
Pesos ajustados para reduzir loss
```

### 1.2 Gradiente Descendente (Gradient Descent)

É o coração do aprendizado em IA. Imagina uma bola rolando downhill em um vale, o "valley" representa o espaço de parâmetros e a altura representa o loss.

**Variantes:**

**Batch Gradient Descent:**
- Calcula gradiente em TODO o dataset
- Atualiza pesos uma vez por época
- Vantagem: Direção estável e acurada
- Desvantagem: Lento, requer muito RAM

**Stochastic Gradient Descent (SGD):**
- Calcula gradiente em UM exemplo
- Atualiza pesos imediatamente
- Vantagem: Rápido, pode escapar mínimos locais
- Desvantagem: Ruidoso, oscilações

**Mini-batch Gradient Descent:**
- Calcula gradiente em PEQUENO LOTE (32-256 exemplos)
- Melhor dos dois mundos
- Mais usado na prática

**Otimizadores Avançados:**

**Momentum:**
```
v = γv + ∂L/∂w  (acumula "velocidade")
w = w - α*v
```
- Continua movimento em direção que estava indo
- Acelera convergência

**Adam (Adaptive Moment Estimation):**
```
Combina momentum com taxa de aprendizado adaptativa
- Cada parâmetro tem sua própria "learning rate"
- Usa primeiro e segundo momentos dos gradientes
```
- Muito popular, geralmente funciona bem

### 1.3 Função de Perda (Loss Function)

Define o que significa estar "errado". Diferentes tarefas usam diferentes perdas:

**Regressão (prever valores contínuos):**

**Mean Squared Error (MSE):**
```
L = (1/N) Σ(ŷᵢ - yᵢ)²
```
- Penaliza erros grandes fortemente (quadrado)
- Sensível a outliers

**Mean Absolute Error (MAE):**
```
L = (1/N) Σ|ŷᵢ - yᵢ|
```
- Mais robusto a outliers
- Menos diferenciável

**Classificação Binária:**

**Binary Cross Entropy:**
```
L = -(1/N) Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```
- Penaliza confiança nas previsões erradas
- Ótimo para calibração probabilística

**Classificação Multiclasse:**

**Categorical Cross Entropy:**
```
L = -(1/N) Σ Σ yᵢc * log(ŷᵢc)
```
- Extensão para múltiplas classes
- Usada com softmax

---

## SEÇÃO 2: ARQUITETURAS CLÁSSICAS

### 2.1 Redes Convolucionais (CNNs)

Projetadas especialmente para processar imagens. A ideia central é que características locais são mais importantes que relações globais.

#### **Componentes Principais**

**Camada Convolucional (Conv Layer):**

```
Uma imagem 5x5:
[1 2 3 0 0]
[4 5 6 0 0]
[7 8 9 0 0]
[0 0 0 1 1]
[0 0 0 1 1]

Um filtro 3x3 (kernel):
[0 -1  0]
[-1 4 -1]
[0 -1  0]  (detecção de bordas)

Convolução no ponto (0,0):
1*0 + 2*(-1) + 3*0 + 4*(-1) + 5*4 + 6*(-1) + 7*0 + 8*(-1) + 9*0 = 20 - 2 - 4 - 8 = 6

Resultado é uma "feature map" mostrado características detectadas pelo filtro
```

**Intuição:**
- Cada filtro detecta um padrão (bordas, texturas, formas)
- O filtro "desliza" pela imagem
- Valores elevados na feature map = padrão presente

**Pooling (Max Pooling):**

```
Feature map 4x4:
[1  3  4  2]
[5  8  2  1]
[2  4  3  6]
[1  2  5  4]

Max Pooling 2x2:
[8  4]  (máximo em cada região 2x2)
[4  6]

Resultado: Reduz dimensionalidade, mantém features importantes
```

**Vantagens de CNNs:**
- Parâmetros compartilhados (mesmo filtro aplicado em vários lugares)
- Invariância a translações pequenas
- Muito mais eficiente que redes totalmente conectadas para imagens

#### **Arquitetura Típica CNN**

```
Entrada (Imagem 28x28x3)
    ↓
Conv (32 filtros 3x3) → 26x26x32
    ↓
ReLU
    ↓
MaxPool (2x2) → 13x13x32
    ↓
Conv (64 filtros 3x3) → 11x11x64
    ↓
ReLU
    ↓
MaxPool (2x2) → 5x5x64
    ↓
Flatten → 1600 valores
    ↓
Dense (128 neurônios) + ReLU
    ↓
Dropout (30%)  [regularização]
    ↓
Dense (10 neurônios) + Softmax
    ↓
Saída (10 classes)
```

### 2.2 Redes Recorrentes (RNNs e LSTMs)

Projetadas para processar sequências. Mantêm um "estado oculto" que passa entre tempo t e t+1.

#### **RNN Básica**

```
Sequência: "Gato sentou na..."

t=0: Input="Gato"  → Hidden State h₀ → Output o₀
t=1: Input="sentou" → Hidden State h₁ → Output o₁  (h₁ usa h₀)
t=2: Input="na"    → Hidden State h₂ → Output o₂  (h₂ usa h₁)

Equação:
hₜ = tanh(Wₕₕ * hₜ₋₁ + Wₓₕ * xₜ + bₕ)
oₜ = Wₕₒ * hₜ + bₒ
```

**Problema: Vanishing/Exploding Gradient**

Em sequências longas:
- Gradientes diminuem exponencialmente (vanishing)
- Ou aumentam exponencialmente (exploding)
- Rede não consegue aprender dependências de longo prazo

#### **LSTM (Long Short-Term Memory)**

Solução para o problema de longo prazo. Adiciona estruturas de "portão" (gates):

```
Célula LSTM tem 4 componentes:

1. Forget Gate (esquecer)
   fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
   Decide o que descartar do estado anterior

2. Input Gate (entrada)
   iₜ = σ(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)
   Decide qual informação nova é importante

3. Candidate Values (candidatos)
   C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
   Novos valores candidatos

4. Output Gate (saída)
   oₜ = σ(Wₒ · [hₜ₋₁, xₜ] + bₒ)
   Decide o que mostrar para próximo estado

Cell State (estado da célula - memória de longo prazo):
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ

Hidden State (saída):
hₜ = oₜ ⊙ tanh(Cₜ)
```

**Por que funciona:**
- Caminho de gradiente mais direto através da célula
- Portões permitem aprender quando guardar/descartar informação
- Pode manter contexto por muitos passos de tempo

#### **Aplicações:**

- Tradução automática (sequence-to-sequence)
- Reconhecimento de fala
- Previsão de série temporal
- Classificação de sentimento em textos

### 2.3 Arquitetura Transformer

Revolucionou o processamento de sequências. Substituiu recorrência por atenção.

#### **O Problema com RNNs:**

- Processamento sequencial: deve-se esperar passo i-1 para processar passo i
- Não paraleliza bem
- Difícil treinar em GPUs/TPUs (há espera)

#### **A Solução: Atenção Própria (Self-Attention)**

Ideia central: Cada elemento da sequência "presta atenção" a todos os outros elementos diretamente.

```
Sequência: "O gato está dormindo"
Palavras:  w₁="O", w₂="gato", w₃="está", w₄="dormindo"

Para w₂="gato", queremos saber:
- Qual informação de w₁ ("O") é relevante?
- Qual informação de w₂ ("gato") é relevante?
- Qual informação de w₃ ("está") é relevante?
- Qual informação de w₄ ("dormindo") é relevante?

Resposta: Calcular "atenção" para cada outro elemento
```

#### **Mecânica de Atenção (Scaled Dot-Product Attention)**

```
Entrada: Sequência de palavras codificadas como vetores

1. Gerar Query (Q), Key (K), Value (V):
   Q = X * Wq  (o que procuramos)
   K = X * Wk  (o que podemos encontrar)
   V = X * Wv  (informação a recuperar)

2. Calcular Similarity (Pontuações de Atenção):
   scores = Q · Kᵀ / √(dₖ)
   
   Exemplo numérico:
   Q="gato" = [0.1, 0.2, 0.3]
   K = [
     [0.2, 0.1, 0.3],  K₁ para "O"
     [0.1, 0.2, 0.3],  K₂ para "gato"
     [0.3, 0.1, 0.2],  K₃ para "está"
     [0.2, 0.3, 0.1]   K₄ para "dormindo"
   ]
   
   scores = [0.06+0.02+0.09, 0.01+0.04+0.09, 0.03+0.02+0.06, 0.02+0.06+0.03]
          = [0.17, 0.14, 0.11, 0.11]

3. Softmax dos Scores (normalizar):
   attention_weights = softmax(scores)
                     ≈ [0.33, 0.27, 0.20, 0.20]
   
   Interpretação: "gato" presta mais atenção a si mesmo (0.33)

4. Multiplicar pelos Values (recuperar informação):
   output = Σ attention_weights[i] * V[i]
          = 0.33*V_gato + 0.27*V_O + 0.20*V_está + 0.20*V_dormindo
```

#### **Multi-Head Attention**

```
Não usa apenas um head de atenção, mas múltiplos:

Head 1: Aprende a prestar atenção a posição anterior
Head 2: Aprende a prestar atenção a posição seguinte
Head 3: Aprende a prestar atenção a palavras semanticamente similares
...

Cada head:
- Tem seus próprios Q, K, V, parâmetros
- Processa em paralelo
- Resultados concatenados

Fórmula:
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ) * Wᵒ
```

**Vantagem:** Cada head aprende aspectos diferentes de como prestar atenção.

#### **Arquitetura Completa do Transformer**

```
Input: Sequência de palavras

1. Embedding + Positional Encoding
   - Converte palavras em vetores
   - Adiciona informação de posição (modelos não sabem ordem naturalmente)

2. Multi-Head Self-Attention
   - Cada elemento interage com todos os outros

3. Feed-Forward Network
   FFN(x) = max(0, x*W₁ + b₁)*W₂ + b₂
   Aplicado elemento por elemento

4. Residual Connections + Layer Normalization
   output = LayerNorm(input + SubLayer(input))
   Facilita treinamento em redes profundas

5. Repetir passos 2-4 múltiplas vezes (N camadas)

Decoder (para geração):
- Similar ao encoder
- Mas masked self-attention (pode ver passado, não futuro)
- Atenção cruzada (entre encoder e decoder)

Saída: Logits para próxima palavra
```

---

## SEÇÃO 3: APRENDIZADO PROFUNDO MODERNO

### 3.1 Embedding e Representação Vetorial

Uma das ideias mais poderosas de ML: representar informação categórica (palavras, usuários, produtos) como vetores numéricos.

#### **Word2Vec: Capturando Semântica**

```
Ideia: Treinar rede neural para prever palavra próxima em sequência

Skip-gram model:
Input: "gato"
Output esperado: ["o", "está", "dormindo"]

Rede neural aprende representações vetoriais (embeddings) que capturam semântica.

Resultado interessante após treinamento:
vec("rei") - vec("homem") + vec("mulher") ≈ vec("rainha")

Interpretação: Diferença semântica entre "rei" e "homem" é próxima a diferença entre "rainha" e "mulher"
```

**Vantagem:** Sem rotulação manual, apenas usando estrutura de linguagem natural.

#### **GloVe: Estatísticas Globais**

Similar ao Word2Vec, mas usa matriz de co-ocorrência:

```
Matriz de co-ocorrência:
              "gato"  "cão"  "feliz"  "triste"
"gato" é        0      5       2        0
"gato" não é    0      1       0        4
"gato" parece   0      1       3        0

Treina embeddings para explicar essas estatísticas globais
```

#### **Contextualized Embeddings (ELMo, BERT)**

Problema com Word2Vec/GloVe: Mesma palavra sempre tem mesmo vetor.

```
Frase 1: "Banco de dados"
Frase 2: "Sentar no banco"

Mesma palavra "banco", significados diferentes!

Solução (ELMo, BERT):
- Processa sequência inteira
- Contexto determina embedding da palavra
- "banco" em contexto de database ≠ "banco" em contexto de móvel
```

### 3.2 Pré-treino e Fine-tuning

A revolução que permitiu a explosão de IA moderna.

#### **Pré-treino em Larga Escala**

```
Tarefa de pré-treino (unsupervised/self-supervised):

1. Language Model: Predizer próxima palavra
   Entrada: "O gato está"
   Saída: predizer "dormindo", "comendo", "correndo"
   
2. Masked Language Model (MLM - BERT):
   Entrada: "O gato está [MASK] na cama"
   Saída: predizer "[MASK]" = "dormindo"
   
3. Next Sentence Prediction (NSP):
   Entrada: "O gato está dormindo. [SEP] Ele está cansado."
   Saída: Predizer se segunda frase segue primeira (SIM/NÃO)

Treina em bilhões de palavras de texto genérico.
```

**Por que funciona:**
- Modelo é forçado a entender semântica, sintaxe, factualidade
- Aprende representações generalizáveis
- Quando fine-tuned em tarefa específica, começa de ponto muito melhor

#### **Fine-tuning para Tarefa Específica**

```
Pré-treino (genérico):
BERT treinado em Wikipedia + Books

Fine-tuning (específico):
1. Carrega pesos pré-treinados
2. Adiciona camadas específicas da tarefa
3. Treina em dataset pequeno da tarefa

Exemplo: Análise de Sentimento
- Coloca token [CLS] no começo
- Saída de [CLS] passa por camada densa
- Treina apenas essa camada + ajustes menores aos pesos pré-treinados

Resultado: Excelente performance com pouco dado
```

**Taxa de Aprendizado no Fine-tuning:**
- Usa learning rate MUITO menor (2e-5 vs 1e-3)
- Protege pesos já treinados
- Apenas ajustes sutis

### 3.3 Função Softmax e Probabilidades

Essencial para classificação.

```
Logits (saídas brutas da rede): [2.0, 1.0, 0.1]

Softmax:
e^2.0 = 7.39, e^1.0 = 2.72, e^0.1 = 1.11
soma = 11.22

Probabilidades: [7.39/11.22, 2.72/11.22, 1.11/11.22]
              = [0.659, 0.242, 0.099]

Propriedade: soma sempre = 1, interpretável como distribuição
```

### 3.4 Regularização: Evitando Overfitting

Modelos podem memorizar dados de treino sem aprender padrões generalizáveis.

#### **Dropout**

```
Durante treino (aleatoriamente):
- 30% dos neurônios são "desligados"
- Força rede a aprender representações redundantes
- Cada neurônio não pode confiar em colegas específicos

Durante teste/inferência:
- Todos neurônios ligados
- Escalados por (1 - dropout_rate)

Efeito: Rede mais robusta, generaliza melhor
```

#### **L1 e L2 Regularização**

```
Perda = Loss_original + λ * Regularização

L2 (Ridge): λ * Σ wᵢ²
- Penaliza pesos grandes
- Força muitos pesos para próximo de zero
- Resultado: Modelo mais "suave"

L1 (Lasso): λ * Σ |wᵢ|
- Pode forçar alguns pesos exatamente a zero
- Seleção de features natural
```

#### **Batch Normalization**

```
Normaliza ativações em cada batch:

1. Calcular média e variância do batch
   μ_batch = (1/m) Σ aᵢ
   σ_batch² = (1/m) Σ (aᵢ - μ_batch)²

2. Normalizar
   a_norm = (a - μ_batch) / √(σ_batch² + ε)

3. Escalar e deslocar (aprendíveis)
   a_final = γ * a_norm + β

Efeito:
- Estabiliza treinamento
- Permite learning rates maiores
- Atua como regularização
```

---

## SEÇÃO 4: ARQUITETURAS MODERNAS ESPECIALIZADAS

### 4.1 Modelos de Linguagem Generativos (GPT)

Fundamentados em "causal language modeling" e transformers.

#### **Causal Language Modeling**

```
Objetivo: Predizer próxima palavra em sequência

Treino:
Input:  "O gato está"
Output: "dormindo"

Input:  "O gato está dormindo"
Output: "na"

Input:  "O gato está dormindo na"
Output: "cama"

Rede aprende: Dado histórico de palavras, qual palavra é mais provável?
```

#### **Temperatura (Temperature Sampling)**

Controla "criatividade" vs "segurança" na geração:

```
Logits brutos: [2.0, 1.0, 0.1]

Com temperatura T=1.0 (normal):
Softmax([2.0, 1.0, 0.1]) = [0.659, 0.242, 0.099]

Com temperatura T=0.5 (mais determinístico):
Softmax([2.0/0.5, 1.0/0.5, 0.1/0.5]) = Softmax([4.0, 2.0, 0.2])
= [0.946, 0.051, 0.003]  ← Muito mais confiante na primeira opção

Com temperatura T=2.0 (mais aleatório):
Softmax([2.0/2.0, 1.0/2.0, 0.1/2.0]) = Softmax([1.0, 0.5, 0.05])
= [0.505, 0.251, 0.244]  ← Mais distribuído, menos previsível
```

**Interpretação:** T baixa = mais "focado", T alta = mais "exploratório"

#### **Top-K e Top-P Sampling**

```
Métodos para evitar palavras improváveis:

Top-K: Apenas amostra das K mais prováveis palavras
Exemplo: K=5
Todas as palavras menos as 5 top têm probabilidade zerada

Top-P (Nucleus Sampling): Amostra das palavras mais prováveis até atingir P% da probabilidade acumulada
Exemplo: P=0.9
Ordena palavras por probabilidade
Mantém apenas palavras até soma = 0.9
```

### 4.2 Attention Mechanisms Avançados

#### **Cross-Attention**

Usado em modelos sequence-to-sequence (tradução, sumarização):

```
Encoder:   Processa sequência de entrada
Query:     Vem do decoder (o que procuramos)
Key/Value: Vem do encoder (onde procuramos)

Exemplo: Tradução "O gato está dormindo" → "Le chat dort"

Decoder gera: "Le"
Query = embedding de "Le"
Usa cross-attention para encontrar qual parte do inglês é relevante
Descobre que precisa de "gato" e "está dormindo"
```

#### **Sparse Attention**

Para sequências muito longas, atenção completa é cara (O(n²)):

```
Janela local: Cada posição atende apenas vizinhos próximos
    [x x x X x x x]  X atende apenas a vizinhos

Atenção em stride: Cada posição atende a cada k-ésima posição
    [x . x . x . x]  Espaçamento reduz complexidade

Atenção de longa distância: Alguns tokens "sink" especiais atendem a tudo

Resultado: O(n) ou O(n log n) vs O(n²)
```

### 4.3 Vision Transformers (ViT)

Aplicar transformers a imagens (não apenas texto):

```
Imagem 224x224x3

1. Dividir em patches 16x16
   (224/16) × (224/16) = 14 × 14 = 196 patches

2. Cada patch linear embedded
   196 vetores de embedding

3. Adicionar [CLS] token no começo
   197 tokens totais

4. Adicionar positional embeddings
   (diferente de sequências: 2D position)

5. Passar por transformer normal
   Self-attention entre patches

Resultado: Pode classificar imagem diretamente com transformers!
```

---

## SEÇÃO 5: TREINAMENTO EM LARGA ESCALA

### 5.1 Computação Distribuída

Treinar modelos gigantes requer múltiplos devices (GPUs/TPUs).

#### **Data Parallelism**

```
Modelo cabe em uma GPU

4 GPUs:
GPU 0: Treina com batch 0
GPU 1: Treina com batch 1
GPU 2: Treina com batch 2
GPU 3: Treina com batch 3

Sincronizar gradientes, atualizar pesos, repetir
```

**Desvantagem:** Comunicação entre devices é gargalo

#### **Model Parallelism**

```
Modelo NÃO cabe em uma GPU

Camadas 1-2:    GPU 0
Camadas 3-4:    GPU 1
Camadas 5-6:    GPU 2
Camadas 7-8:    GPU 3

Ativa camadas sequencialmente (pipeline)
Mais complexo, menos overhead de comunicação
```

#### **Pipeline Parallelism**

Hibrido que evita que GPUs fiquem ociosas:

```
Tempo 1: GPU0 processa micro-batch 0
Tempo 2: GPU0 processa micro-batch 1, GPU1 processa saída de batch 0
Tempo 3: GPU0 processa micro-batch 2, GPU1 processa saída de batch 1, GPU2 processa saída do GPU1
...
Resultado: Melhor utilização de recursos
```

### 5.2 Quantização (Model Compression)

Reduzir tamanho do modelo para inferência mais rápida:

#### **FP32 → FP16**

```
FP32 (float32): 32 bits por número
Gama: ~10^-38 a 10^38
Precisão: 7 dígitos decimais

FP16 (float16): 16 bits por número
Gama: ~10^-4 a 10^4
Precisão: 3-4 dígitos decimais

Redução de tamanho: 2x
Speedup: Até 10x em GPUs modernas

Problema: Underflow, gradientes muito pequenos
Solução: Loss scaling (multiplicar loss por fator grande, gradientes maiores)
```

#### **INT8 (Quantização para Inteiros)**

```
FP32: 4 bytes
INT8: 1 byte
Redução: 4x de tamanho

Quantização linear:
INT8 = round((FP32 - min_value) / (max_value - min_value) * 255)

Desvantagem: Perda de precisão mais significativa
Usado para: Inferência apenas (não treino)
```

#### **Knowledge Distillation**

```
Treinar modelo pequeno para imitar modelo grande:

Teacher (grande):
- 175 bilhões de parâmetros (GPT-3)
- Saídas "soft" (distribuições de probabilidade)

Student (pequeno):
- 1 bilhão de parâmetros
- Treinado em:
  - Dados originais
  - MAIS saídas do teacher (soft targets)

Resultado: Student aprende padrões do teacher
Menor que teacher mas muito melhor que treinar do zero
```

### 5.3 Normalização de Aprendizado (Learning Rate Scheduling)

Taxa de aprendizado fixa não é ótima:

```
Learning Rate muito alto:
- Pula mínimo ótimo
- Treino oscila ou diverge

Learning Rate muito baixo:
- Converge lentamente
- Pode ficar preso em mínimos locais

Solução: Variar learning rate durante treino
```

#### **Warm-up Linear Decay**

```
Epochs 0-1000:      Learning rate aumenta linearmente (warm-up)
Epochs 1000-10000:  Learning rate diminui linearmente (decay)

Intuição:
- Começo: Pesos inicializados aleatoriamente, precisa "acertar"
- Fim: Sintonização fina perto de mínimo ótimo
```

#### **Cosine Annealing**

```
Learning rate segue coseno:
lr(t) = 0.5 * lr_max * (1 + cos(πt/T))

Começa alto, cai suavemente para zero
Evita quedas abruptas
```

---

## SEÇÃO 6: INFERÊNCIA E DEPLOYMENT

### 6.1 Batching para Throughput

Processar múltiplos exemplos em paralelo:

```
Um exemplo por vez:
Tempo: 10ms por exemplo
Throughput: 100 exemplos/segundo

Batch de 32 exemplos:
Tempo: 250ms para 32 exemplos
Throughput: 128 exemplos/segundo (28% mais rápido!)

Razão: GPUs são melhores com operações em larga escala
Overhead de lançamento é amortizado
```

### 6.2 Quantização na Inferência

Reduzir latência e uso de memória:

```
Modelo original (FP32): 350MB em memória
Modelo quantizado (INT8): 87MB em memória
Speedup de inferência: ~2-4x

Tradeoff: Pequena perda de acurácia (geralmente <1%)
```

### 6.3 Caching e KV-Cache

Para modelos autoregressivos (gerando uma palavra por vez):

```
Primeiro token (input_ids=[101, 2054, 2003]):
Computa Q, K, V para todos os tokens
Atenção entre todos
Gera predição para próximo token

Segundo token (apenas novo token):
NÃO precisa recomputar atenção do primeiro token
Apenas computa novo K, V
Reutiliza K, V antigos (KV-cache)

Resultado: Cada novo token 10-100x mais rápido!
```

---

## SEÇÃO 7: CONCEITOS AVANÇADOS

### 7.1 Interpretabilidade: Visualizando o que IA Aprende

#### **Atenção Visualizada**

```
Frase: "O gato viu o rato"

Para palavra "gato", mapa de atenção:
"O":    ▓░░░░░ 15%
"gato": ▓▓▓▓▓░ 85%
"viu":  ░░░░░░ 0%
"o":    ░░░░░░ 0%
"rato": ░░░░░░ 0%

Interpretação: "gato" presta atenção principalmente a si mesmo
```

#### **Saliência (Gradient-based)**

```
Qual pixel da imagem mais afeta a previsão?

Calcula: ∂(output)/∂(input)
Pixels com gradiente alto = importantes

Visualização: Mapa de calor sobre imagem
Brilho indica importância
```

#### **Probing Tasks**

```
Hipótese: BERT aprendeu estrutura gramatical?

Teste: Usar hidden layer de BERT para:
- Predizer part-of-speech (verbo, nome, etc)
- Se consegue fazer bem → aprendeu gramaticidade
```

### 7.2 Alucinações em LLMs

Fenômeno onde modelo gera fatos inventados confientemente.

```
Entrada: "Qual é o nome do primeiro presidente do Brasil?"
Saída (confiante): "João da Silva"
Realidade: "Deodoro da Fonseca" OU "Manuel Deodoro da Fonseca"

Por que acontece?
1. Modelo foi treinado em muito texto com erros
2. Distribuição de treino favorece respostas "que soam bem"
3. Tokenização e embeddings não garantem factualidade
4. Modelo não tem acesso a base de conhecimento em tempo real
```

#### **Mitigação com RAG (Retrieval-Augmented Generation)**

```
Abordagem tradicional LLM:
Pergunta → LLM → Resposta (pode alucinar)

Abordagem RAG:
Pergunta → Recuperar documentos relevantes → LLM com contexto → Resposta

Exemplo:
Pergunta: "Quem é o presidente?"
Recuperar: Artigo Wikipedia sobre "Presidente da República"
Prompt: "De acordo com [documento], quem é..."
Resposta: Baseada em fato recuperado (menos alucinação)
```

### 7.3 Alinhamento e RLHF

Como fazer LLMs seguir instruções e evitar comportamentos indesejados?

#### **Supervised Fine-Tuning (SFT)**

```
Base: GPT-3 (gerador de linguagem)
Problema: Pode gerar conteúdo tóxico, desinformação, etc

Solução 1: Fine-tune em exemplos de bom comportamento
Dataset curado: Instruções + Respostas boas
Treina modelo a seguir instruções

Mas: Qual definição de "bom"? Difícil de especificar
```

#### **RLHF (Reinforcement Learning from Human Feedback)**

```
Passo 1: Supervised Fine-Tuning
- Treina em exemplos de boa qualidade

Passo 2: Colecionar Preferências Humanas
- Mesma instrução, múltiplas respostas modelo
- Humanos escolhem: Resposta A > Resposta B
- Dataset de preferências

Passo 3: Treinar Reward Model
- Aprende: Resposta A é melhor que Resposta B
- Essencialmente: modelo que classifica qualidade

Passo 4: Otimizar com Policy Gradient
- Usa reward model para guiar RL
- Modelo LLM aprende a maximizar reward
- Sem dados de treino, apenas feedback

Resultado: ChatGPT é muito mais útil e seguro que GPT-3!
```

---

## SEÇÃO 8: APLICAÇÕES ESPECÍFICAS

### 8.1 Visão Computacional: Detecção de Objetos (YOLO)

```
Tarefa: Identificar TODOS os objetos em uma imagem com caixas

Abordagem ingênua: Redes convolucionais tradiconais
- Problema: Encontrar objeto, não localizar

YOLO (You Only Look Once):
1. Divide imagem em grid (13x13, 26x26 etc)
2. Cada célula prediz:
   - Probabilidade de objeto estar ali
   - Coordenadas da caixa (x, y, width, height)
   - Classe (carro, pessoa, etc)

3. Pós-processamento:
   - Remove caixas com confiança baixa
   - Non-maximum suppression: remove caixas sobrepostas

Vantagem: Rápido, em tempo real
Desvantagem: Pode perder objetos pequenos
```

### 8.2 Processamento de Linguagem Natural: Análise de Sentimento

```
Tarefa: Classificar se review é positivo, negativo ou neutro

Abordagem moderna (BERT):
1. Tokenizar texto
2. Passar por BERT pré-treinado
3. Pegar saída do token [CLS] (representa documento)
4. Passar por camada densa simples
5. Softmax para classe

Treino:
- Fine-tune em dataset pequeno (500-1000 reviews)
- BERT mantém pesos principalmente, ajusta levemente
- Learning rate muito pequeno

Resultado: ~95% acurácia com pouco dado!
```

### 8.3 Transformação de Imagem: Style Transfer

```
Objetivo: Pegar imagem de conteúdo + estilo de referência
Gerar: Imagem de conteúdo no estilo da referência

Abordagem com Redes Neurais:

1. Carregar VGG (rede convolucional pré-treinada)
2. Definir imagem de saída como variável otimizável

3. Loss total = Content Loss + Style Loss
   
   Content Loss:
   - Compara features de camadas profundas
   - Imagem otimizada deveria ter mesmo conteúdo
   
   Style Loss:
   - Compara Gram matrices de features
   - Gram matrix: correlações entre features
   - Captura "textura" e "estilo"

4. Otimiza imagem usando gradient descent
   ∂Loss/∂imagem mostra como ajustar pixels

Iterações: Imagem gradualmente se parece com conteúdo + estilo
```

---

## CONCLUSÃO

Os conceitos técnicos de IA não são magia, mas aplicações elegantes de:
- **Cálculo diferencial** (gradientes, backprop)
- **Álgebra linear** (multiplicação de matrizes, transformações)
- **Estatística** (probabilidades, distribuições)
- **Otimização** (encontrar mínimos)

Modernos sistemas de IA como ChatGPT são construções elaboradas, mas fundamentadas nestes princípios. Compreender "por debaixo dos panos" permite não apenas usar IA, mas crítica e responsavelmente avaliar suas capacidades, limitações e implicações éticas.

