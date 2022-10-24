import os
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import nltk  # Here to have a nice missing dependency error message early on
import evaluate
import numpy as np

from transformers import (
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
    DataCollatorForSeq2Seq,
)

from optimum.habana import GaudiSeq2SeqTrainer, GaudiSeq2SeqTrainingArguments
from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments used for the training
    """

    dataset_id: str = field(
        default=None, metadata={"help": "The repository id of the dataset to use (via the datasets library)."}
    )
    model_id: str = field(default=None, metadata={"help": "The repository id of the model to use (via AutoModel)."})
    repository_id: str = field(
        default=None,
        metadata={"help": "The repository id where the model will be saved or loaded from for futher pre-training."},
    )
    hf_hub_token: str = field(
        default=False,
        metadata={"help": "The Token used to push models, metrics and logs to the Hub."},
    )
    gaudi_config_id: Optional[str] = field(
        default=None,
        metadata={"help": "Habana config used for fp16 ops.  more here: https://huggingface.co/Habana"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The Batch Size per HPU used during training"},
    )
    num_epochs: Optional[int] = field(
        default=1_000_000,
        metadata={"help": "The Number of Training steps to perform."},
    )
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "Learning Rate for the training"})
    deepspeed: str = field(
        default=None,
        metadata={"help": "Path to the deepspeed config file."},
    )


def run_mlm():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Script parameters {script_args}")

    # set seed for reproducibility
    seed = 34
    set_seed(seed)

    # load processed dataset
    dataset = load_dataset(script_args.dataset_id)
    # load trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_auth_token=script_args.hf_hub_token)

    max_input_length = 512
    max_target_length = 64
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + txt for txt in examples["text"]]

        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["title"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # run processing
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_id)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # define our hyperparameters
    gaudi_training_args = GaudiSeq2SeqTrainingArguments(
        output_dir=script_args.repository_id,
        use_habana=True,
        use_lazy_mode=True,
        deepspeed=script_args.deepspeed,
        gaudi_config_name=script_args.gaudi_config_id,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        learning_rate=script_args.learning_rate,
        seed=seed,
        num_train_epochs=script_args.num_epochs,
        # logging & evaluation strategies
        logging_dir=f"{script_args.repository_id}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="steps",
        save_steps=5_000,
        save_total_limit=2,
        report_to="tensorboard",
        # push to hub parameters
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=script_args.repository_id,
        hub_token=script_args.hf_hub_token,
    )

    # Initialize our Trainer
    trainer = GaudiSeq2SeqTrainer(
        model=model,
        args=gaudi_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # train the model
    trainer.train()


if __name__ == "__main__":
    run_mlm()
