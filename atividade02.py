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

print("=== Processamento de Linguagem Natural ===")
print(f"Frase original: '{frase_exemplo}'")
print(f"Tokens (IDs): {tokens_ids}")
print("\n--- Tabela de Vocabulário ---")
print(vocabulario_df.to_string(index=False))

print("\n--- Dimensões do Processamento ---")
print(f"Tabela de Embeddings (Vocabulário x Dimensão): {matriz_identidade_visual.shape}")
print(f"Tensor Final X (Batch, Sequência, Dimensão): {X.shape}")