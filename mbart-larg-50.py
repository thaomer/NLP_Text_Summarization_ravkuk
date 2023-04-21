from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import math
import numpy as np


ravkuk_dataset = load_dataset("thaomer/ravkuk_summerize_dataset", split="train")
ravkuk_dataset = ravkuk_dataset.train_test_split(test_size=0.1)

model_checkpoint = "facebook/mbart-large-50"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 800
max_target_length = 90


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["summary"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = ravkuk_dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


batch_size = 8
num_train_epochs = 10

args = Seq2SeqTrainingArguments(
    output_dir="./le-fine-tune-mbart-large-50",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy='epoch',
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=num_train_epochs,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    push_to_hub=False,
)

metric = evaluate.load("rouge")


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, tokenizer=lambda x: x.split())

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(
    ravkuk_dataset["train"].column_names
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()