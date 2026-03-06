import pandas as pd
import numpy as np

mapeamento_palavras = {"gabriel": 0, "fez": 1, "a": 2, "atividade": 3}
vocabulario_df = pd.DataFrame(list(mapeamento_palavras.items()), columns=['Palavra', 'ID'])

tamanho_vocabulario = len(mapeamento_palavras)
d_model = 64  

frase_exemplo = "gabriel fez a atividade"
tokens_ids = [mapeamento_palavras[palavra] for palavra in frase_exemplo.split()]

np.random.seed(10)
matriz_identidade_visual = np.random.randn(tamanho_vocabulario, d_model)

entrada_embeddings = matriz_identidade_visual[tokens_ids] 

X = np.expand_dims(entrada_embeddings, axis=0) 

print("\n--- Tabela de Vocabulário ---")
print(vocabulario_df.to_string(index=False))

print("\n--- Dimensões do Processamento ---")
print(f"Tabela de Embeddings (Vocabulário x Dimensão): {matriz_identidade_visual.shape}")
print(f"Tensor Final X (Batch, Sequência, Dimensão): {X.shape}")

def aplicar_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def camada_normalizacao(x, epsilon=1e-6):
    media = np.mean(x, axis=-1, keepdims=True)
    variancia = np.var(x, axis=-1, keepdims=True)
    return (x - media) / np.sqrt(variancia + epsilon)

def mecanismo_atencao(X, d_model):
    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)
    
    Q = X @ Wq  
    K = X @ Wk  
    V = X @ Wv  
    
    K_t = np.transpose(K, axes=(0, 2, 1))
    scores = (Q @ K_t) / np.sqrt(d_model)
    
    pesos = aplicar_softmax(scores)
    return pesos @ V

def rede_feed_forward(x, d_model, d_ff=256):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    
    intermediario = np.maximum(0, x @ W1 + b1)
    return intermediario @ W2 + b2

N = 6 
d_model = 64 

print(f"--- Iniciando o Encoder Stack (N={N}) ---")

for i in range(N):
    X_att = mecanismo_atencao(X, d_model)
    X_norm1 = camada_normalizacao(X + X_att)
    X_ffn = rede_feed_forward(X_norm1, d_model)
    X_out = camada_normalizacao(X_norm1 + X_ffn)
    X = X_out
    
    print(f"Camada {i+1}: Shape mantido em {X.shape}")

Vetor_Z = X
print(f"\nValidação do Shape final do Vetor Z: {Vetor_Z.shape}")
print(f"Amostra dos primeiros 5 valores (features) da primeira palavra:\n{Vetor_Z[0, 0, :5]}")