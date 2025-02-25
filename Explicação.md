# Destilação de Modelos

> Modelo Professor:<br> 
> Modelo Aluno: 

---
### Objetivo:
- Este código tem como objetivo realizar a distilação de conhecimento de um modelo maior (BERT) para um modelo menor (DistilBERT) utilizando o dataset LIAR, que é utilizado para classificar declarações como verdadeiras, falsas ou sem base para uma conclusão.

- A técnica de distilação de conhecimento permite que um modelo menor aprenda a partir das previsões de um modelo maior, mantendo uma boa performance com menos parâmetros.

---
### Bibliotecas

- **torch**: Utilizado para treinamento, calcular a perda de informação e otimizar os parâmetros do modelo.

- **transformers**: A biblioteca `transformers` da Hugging Face contém modelos prontos de NLP (Natural Language Processing) como BERT e DistilBERT, além de fornecer ferramentas para tokenização e treinamento de modelos.

- **datasets**: A biblioteca `datasets` da Hugging Face permite o fácil acesso e manipulação de datasets de NLP. No nosso caso, estamos usando o dataset LIAR, que contém declarações politicas que são classificadas como verdadeiras ou falsas.


---
## Carregando os Modelos:

- **teacher_model**: O modelo professor é o BERT, um modelo de linguagem grande. Ele é carregado usando a função `BertForSequenceClassification.from_pretrained()`, onde o modelo foi pré-treinado no dataset "uncased", ou seja, sem diferenciação entre maiúsculas e minúsculas.

- **student_model**: O modelo aluno é o DistilBERT, uma versão compacta e mais eficiente do BERT. Ele também é carregado da mesma forma que o modelo BERT, mas com menos parâmetros.

---
## Carregando o Tokenizer:

- **tokenizer**: O tokenizer é utilizado para transformar o texto bruto em tokens que o modelo pode entender. O `DistilBertTokenizer` é o tokenizer associado ao modelo DistilBERT. Ele transforma o texto de entrada em tokens, que são então convertidos para IDs que o modelo pode processar. A tokenização é importante para que o modelo consiga entender e processar as palavras de maneira eficiente.

---
## Carregando o Dataset LIAR:
- O dataset LIAR é carregado usando a função `load_dataset("ucsbnlp/liar")` da biblioteca `datasets`. Este dataset contém afirmações de políticos, que são rotuladas como "verdadeiras", "falsas", "majoritariamente verdadeiras", "majoritariamente falsas", "sem base para conclusão" etc.

- As declarações do dataset são passadas por um pré-processamento para garantir que os textos sejam tokenizados e estruturados corretamente para o treinamento.

---
## Função de Pré-processamento:
- A função `preprocess_function` é definida para tokenizar as declarações do dataset, limitando o comprimento dos textos para 128 tokens. O tokenizador converte o texto em IDs que podem ser processados pelo modelo.

---
## Função de Perda de Distilação:
- A função `distillation_loss` calcula a perda combinada de distilação e de classificação. 

- **Distilação**: A perda de distilação utiliza a divergência de Kullback-Leibler (KL) para comparar as distribuições de probabilidade entre o modelo aluno e o modelo professor.

- **Perda de Classificação**: Além disso, a função também calcula a perda de entropia cruzada entre os logits do aluno e os rótulos reais.

- O objetivo é minimizar a diferença entre as previsões do modelo aluno e do professor, além de melhorar a precisão do modelo aluno em relação aos rótulos reais.

---
## Modificando o Trainer:
- O `Trainer` é uma classe fornecida pela Hugging Face para treinar modelos. A classe `DistillationTrainer` herda de `Trainer` e sobrescreve a função `compute_loss` para incluir o cálculo da perda de distilação, utilizando o modelo professor e o modelo aluno.

---
## Inicializando o Treinamento:
- O `DistillationTrainer` é inicializado com os modelos, o dataset de treinamento e de avaliação, e os parâmetros de treinamento.

- O treinamento do modelo é iniciado com a função `trainer.train()`, que vai otimizar os parâmetros do modelo aluno com base na distilação do modelo professor.

---
## Explicando os Conceitos:

- **Softmax**: A função Softmax é usada para transformar os logits de um modelo (os valores brutos antes da normalização) em probabilidades. Isso é essencial para a comparação entre as previsões do aluno e do professor.

- **KL Divergence**: A divergência de Kullback-Leibler é uma medida de diferença entre duas distribuições de probabilidade. No contexto da distilação, ela é usada para calcular a diferença entre as distribuições de probabilidade das previsões do aluno e do professor.

---
## Conclusão
- Com esse código, conseguimos treinar um modelo mais leve, o DistilBERT, para classificar declarações políticas, enquanto aproveitamos o conhecimento de um modelo maior e mais preciso, o BERT.
