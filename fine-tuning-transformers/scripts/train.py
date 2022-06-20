
# TODO: create script which works with GPU and HPU


# from transformers import AutoModelForSequenceClassification,DataCollatorWithPadding
# from optimum.habana import GaudiTrainer, GaudiTrainingArguments
# from huggingface_hub import HfFolder

# # create label2id, id2label dicts for nice outputs for the model
# labels = tokenized_datasets["train"].features["label"].names
# num_labels = len(labels)
# label2id, id2label = dict(), dict()
# for i, label in enumerate(labels):
#     label2id[label] = str(i)
#     id2label[str(i)] = label


# # define training args
# training_args = GaudiTrainingArguments(
#     output_dir=repository_id,
#     use_habana=True,
#     use_lazy_mode=True,
#     gaudi_config_name=gaudi_config_id,
#     num_train_epochs=5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     learning_rate=3e-5,
#     seed=seed,
#     # logging & evaluation strategies
#     logging_dir=f"{repository_id}/logs",
#     logging_strategy="epoch",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     report_to="tensorboard",
#     # push to hub parameters
#     push_to_hub=True,
#     hub_strategy="every_save",
#     hub_model_id=repository_id,
#     hub_token=HfFolder.get_token()
# )

# # define model
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_id,
#     num_labels=num_labels, 
#     id2label=id2label,
#     label2id=label2id,
# )

# if __name__ == "main":
  