# Bibliotecas necessárias
import torch
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, DistilBertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


# Carregar o modelo professor (BERT maior)
teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# Carregar o modelo aluno (DistilBERT)
student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


# Carregar o tokenizer do DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Carregar o dataset LIAR do Hugging Face
dataset = load_dataset("ucsbnlp/liar")


# Função de pré-processamento para tokenizar o texto
def preprocess_function(examples):
    return tokenizer(examples['statement'], truncation=True, padding="max_length", max_length=128)


# Aplicar a tokenização no dataset
train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["test"].map(preprocess_function, batched=True)


# Mapear as classes para 0 (falsa) e 1 (verdadeira)
def map_labels(example):
    if example["label"] in [0, 1, 2]:  # true, mostly_true, half_true
        example["label"] = 1  # Verdadeira
    else:
        example["label"] = 0  # Falsa
    return example

train_dataset = train_dataset.map(map_labels)
eval_dataset = eval_dataset.map(map_labels)


# Definir os parâmetros de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)


# Função de perda para distilação
def distillation_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.5, true_labels=None):
    """
    Calcula a perda de distilação. A perda combina a perda entre os rótulos reais e as previsões
    do professor e as previsões do aluno.
    """
    # Cálculo da perda de distilação usando Softmax com temperatura
    soft_teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
    
    # Cálculo da perda entre as previsões do aluno e do professor (usando Kullback-Leibler Divergence)
    distillation_loss = torch.nn.functional.kl_div(soft_student_probs.log(), soft_teacher_probs, reduction='batchmean')
    
    # A perda total é uma combinação da perda de distilação e a perda de classificação tradicional
    if true_labels is not None:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(student_logits, true_labels)
        return alpha * distillation_loss + (1 - alpha) * cross_entropy_loss
    else:
        return distillation_loss


# Modificar a classe Trainer para incluir o modelo professor
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Passo pelo modelo aluno
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Passo pelo modelo professor
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Calcular a perda de distilação, incluindo a perda de classificação tradicional
        loss = distillation_loss(student_logits, teacher_logits, true_labels=inputs["label"])
        
        return (loss, student_outputs) if return_outputs else loss


# Inicializar o Trainer com o modelo professor
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,  # Seu dataset de treinamento
    eval_dataset=eval_dataset,    # Seu dataset de avaliação
    teacher_model=teacher_model,  # O modelo professor
)


# Treinar o modelo aluno
trainer.train()
