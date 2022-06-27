import argparse
from dataclasses import dataclass
import sys
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict,load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoTokenizer
from huggingface_hub import HfFolder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--use_habana", type=bool)
    args, _ = parser.parse_known_args()

    # training arguments 
    model_id = "xlm-roberta-large"
    gaudi_config_id= "Habana/roberta-large" # more here: https://huggingface.co/Habana
    dataset_id = "AmazonScience/massive"
    dataset_configs=["en-US","de-DE","fr-FR","it-IT","pt-PT","es-ES","nl-NL"]
    seed=33
    repository_id = "habana-xlm-r-large-amazon-massive" if args.use_habana else "gpu-xlm-roberta-large-amazon-massive"
   
    @dataclass
    class hyperparameters:
        num_train_epochs=5
        per_device_train_batch_size=8 if args.use_habana else 1
        per_device_eval_batch_size=8 if args.use_habana else 1
        learning_rate=3e-5
        
    #
    # data preprocessing
    #

    # the columns we want to keep in the dataset
    keep_columns = ["utt", "scenario"]

    # process individuell datasets
    proc_lan_dataset_list=[]
    for lang in dataset_configs:
        # load dataset for language
        lang_ds = load_dataset(dataset_id, lang)
        # only keep the 'utt' & 'scenario column
        lang_ds = lang_ds.remove_columns([col for col in lang_ds["train"].column_names if col not in keep_columns])  
        # rename the columns to match transformers schema
        lang_ds = lang_ds.rename_column("utt", "text")
        lang_ds = lang_ds.rename_column("scenario", "label")
        proc_lan_dataset_list.append(lang_ds)
        
    # concat single splits into one
    train_dataset = concatenate_datasets([ds["train"] for ds in proc_lan_dataset_list])
    eval_dataset = concatenate_datasets([ds["validation"] for ds in proc_lan_dataset_list])
    # create datset dict for easier processing
    dataset = DatasetDict(dict(train=train_dataset,validation=eval_dataset))

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def process(examples):
        tokenized_inputs = tokenizer(
          examples["text"], padding="max_length", truncation=True
        )
        return tokenized_inputs

    tokenized_datasets = dataset.map(process, batched=True)
    tokenized_datasets["train"].features

    # define metrics and metrics function
    f1_metric = load_metric("f1")
    accuracy_metric = load_metric( "accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")
        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"],
        }
  
    # Prepare model labels - useful in inference API
    labels = train_dataset.features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    #
    # model preperation
    #

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels, 
        id2label=id2label,
        label2id=label2id,
    )
    
    
    if args.use_habana == True:
      from optimum.habana import GaudiTrainer, GaudiTrainingArguments
      print("running training on Habana")
      training_args = GaudiTrainingArguments(
          output_dir=repository_id,
          use_habana=True,
          use_lazy_mode=True,
          gaudi_config_name=gaudi_config_id,
          num_train_epochs=hyperparameters.num_train_epochs,
          per_device_train_batch_size=hyperparameters.per_device_train_batch_size,
          per_device_eval_batch_size=hyperparameters.per_device_eval_batch_size,
          learning_rate=hyperparameters.learning_rate,
          seed=seed,
          # logging & evaluation strategies
          logging_dir=f"{repository_id}/logs",
          logging_strategy="epoch",
          evaluation_strategy="epoch",
          save_strategy="epoch",
          save_total_limit=2,
          load_best_model_at_end=True,
          metric_for_best_model="f1",
          report_to="tensorboard",
          # push to hub parameters
          push_to_hub=True,
          hub_strategy="every_save",
          hub_model_id=repository_id,
          hub_token=HfFolder.get_token()
      )
      # create Trainer
      trainer = GaudiTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
      )
    else:
      from transformers import Trainer, TrainingArguments
      print("running training on GPU")
      training_args = TrainingArguments(
          output_dir=repository_id,
          num_train_epochs=hyperparameters.num_train_epochs,
          per_device_train_batch_size=hyperparameters.per_device_train_batch_size,
          per_device_eval_batch_size=hyperparameters.per_device_eval_batch_size,
          learning_rate=hyperparameters.learning_rate,
          seed=seed,
          # logging & evaluation strategies
          logging_dir=f"{repository_id}/logs",
          logging_strategy="epoch",
          evaluation_strategy="epoch",
          save_strategy="epoch",
          save_total_limit=2,
          load_best_model_at_end=True,
          metric_for_best_model="f1",
          report_to="tensorboard",
          # GPU specific
          fp16=True,
          # push to hub parameters
          push_to_hub=True,
          hub_strategy="every_save",
          hub_model_id=repository_id,
          hub_token="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"
      )
      # create Trainer
      trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
      )

    # start training on 1x HPU
    trainer.train()
    # evaluate model
    trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    # create model card
    trainer.create_model_card()