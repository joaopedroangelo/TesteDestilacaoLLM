# Destilação de Modelos

> Modelo Professor: sentence-transformers/paraphrase-mpnet-base-v2<br><br> 
> Modelo Aluno: sentence-transformers/paraphrase-MiniLM-L3-v2

---
### Objetivo:
- Este código tem como objetivo realizar a distilação de conhecimento de um modelo maior para um modelo menor utilizando o dataset AG News, que contém notícias classificadas em quatro categorias: Mundo, Esportes, Negócios e Ciência/Tecnologia.

- A técnica de destilação de conhecimento permite que um modelo menor aprenda a partir das previsões de um modelo maior, mantendo uma boa performance com menos parâmetros e reduzindo os requisitos computacionais para inferência.

---
### Bibliotecas

- **SetFit**: Biblioteca otimizada para classificação de textos baseada em embeddings de frases, permitindo o treinamento eficiente de modelos de classificação sem a necessidade de ajustes finos demorados.

- **Datasets**: A biblioteca `datasets` da Hugging Face permite o fácil acesso e manipulação de conjuntos de dados NLP de forma eficiente.

- **Torch**: Utilizada para gerenciamento do treinamento, cálculo da perda e otimização dos parâmetros do modelo.

- **Transformers**: Biblioteca da Hugging Face que fornece modelos pré-treinados e tokenizadores para tarefas de NLP.

---
## Hardware e Uso de Memória

### **Requisitos de Hardware**

- **Treinamento na GPU:** Recomenda-se pelo menos uma GPU com 8GB de VRAM para treinamento eficiente, especialmente para o modelo professor (`mpnet-base-v2`).

- **Treinamento na CPU:** Possível, mas significativamente mais lento. O modelo aluno (`MiniLM-L3-v2`) pode ser treinado de maneira mais viável na CPU devido ao seu tamanho reduzido.

- **Memória RAM:** Pelo menos 8GB recomendados para evitar gargalos no processamento dos dados.

---
### **Gerenciamento de Memória**

- Modelos maiores podem gerar erros de `CUDA out of memory`, especialmente em GPUs menores. Para mitigar:
  - Reduza o `batch_size`.
  - Utilize modelos mais leves como professor.
  - Libere memória da GPU com `torch.cuda.empty_cache()`.
  - Force o treinamento na CPU se necessário.

---
## Carregando os Modelos:

- **Modelo Professor:** `paraphrase-mpnet-base-v2`, um modelo avançado de embeddings de frases, utilizado para fornecer previsões que servirão de base para o treinamento do modelo aluno.

- **Modelo Aluno:** `paraphrase-MiniLM-L3-v2`, uma versão compacta do modelo professor, treinado para imitar seu comportamento com menos parâmetros e maior eficiência computacional.

---
## Carregando o Dataset AG News:

- O dataset AG News é carregado usando a função `load_dataset("ag_news")`. Ele contém notícias classificadas em quatro categorias.
- Criamos um conjunto reduzido de treino para o modelo professor e um dataset de avaliação.
- Também geramos um dataset não rotulado para a destilação do modelo aluno.

---
## Configuração do Treinamento:

- **Treinamento do Professor:** Utilizamos o `Trainer` da biblioteca SetFit para treinar o modelo professor com um subconjunto do dataset.

- **Treinamento do Aluno:** Após o treinamento do professor, utilizamos a classe `DistillationTrainer`, que permite transferir conhecimento do modelo maior para o menor utilizando os dados não rotulados.

---
## Inicializando o Treinamento:

- O `DistillationTrainer` é inicializado com os modelos, os datasets e os parâmetros de treinamento.

- O treinamento do modelo aluno é realizado com `trainer.train()`, otimizando seus pesos para aproximar suas previsões às do modelo professor.

---
## Conclusão

- Nosso objetivo é ter como resultado um modelo eficiente em termos de memória e tempo de inferência, adequado para aplicações em dispositivos com recursos limitados.
