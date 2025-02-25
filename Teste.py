# Leia: https://huggingface.co/docs/setfit/en/how_to/knowledge_distillation

import torch
from datasets import load_dataset
from setfit import sample_dataset, SetFitModel, TrainingArguments, Trainer, DistillationTrainer


# Limpar cache
torch.cuda.empty_cache()
device = torch.device("cpu")


# Carregar dataset de notícias
print("Carregando dataset...")
dataset = load_dataset("ag_news")


# Criar um conjunto de treino pequeno
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=16)
eval_dataset = dataset["test"]


# Criar um dataset não rotulado para destilação
unlabeled_train_dataset = dataset["train"].shuffle(seed=42).select(range(500))
unlabeled_train_dataset = unlabeled_train_dataset.remove_columns("label")


# Definir modelo professor (mais robusto)
print("Carregando modelo professor...")
teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
teacher_model.to(device)

teacher_args = TrainingArguments(
    batch_size=1,
    num_epochs=2,
)

teacher_trainer = Trainer(
    model=teacher_model,
    args=teacher_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


# Treinar modelo professor
print("Treinando modelo professor...")
teacher_trainer.train()
teacher_metrics = teacher_trainer.evaluate()
print("Métricas do modelo professor:", teacher_metrics)


# Definir modelo aluno (mais leve)
print("Carregando modelo aluno...")
student_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
student_model.to(device)


# Definir parâmetros da destilação
distillation_args = TrainingArguments(
    batch_size=1,
    max_steps=500,
)


# Configurar a destilação
distillation_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    args=distillation_args,
    train_dataset=unlabeled_train_dataset,
    eval_dataset=eval_dataset,
)


# Treinar modelo aluno com destilação
print("Treinando modelo aluno com destilação...")
distillation_trainer.train()
distillation_metrics = distillation_trainer.evaluate()
print("Métricas do modelo aluno:", distillation_metrics)
