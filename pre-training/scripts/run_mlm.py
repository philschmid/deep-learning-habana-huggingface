import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, AutoModelForMaskedLM, AutoTokenizer,set_seed,AutoConfig,DataCollatorForLanguageModeling
from optimum.habana import GaudiTrainer, GaudiTrainingArguments
from datasets import load_dataset, load_metric


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@dataclass
class ScriptArguments:
    """
    Arguments which aren't included in the GaudiTrainingArguments
    """
    dataset_id: str = field(
        default=None, metadata={"help": "The repository id of the dataset to use (via the datasets library)."}
    )
    repository_id: str = field(
        default=None, metadata={"help": "The repository id where the model will be saved or loaded from for futher pre-training."}
    )
    hf_hub_token: str = field(
        default=False,
        metadata={"help": "The Token used to push models, metrics and logs to the Hub."},
    )
    model_config_id: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    gaudi_config_id: Optional[float] = field(
        default="Habana/bert-base-uncased", metadata={"help": "Habana config used for fp16 ops.  more here: https://huggingface.co/Habana"}
    )

    
def run_mlm():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, GaudiTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # set seed for reproducibility
    set_seed(training_args.seed)
    
    # load processed dataset
    train_dataset = load_dataset(script_args.dataset_id)

    # load trained tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(script_args.repository_id, use_auth_token=script_args.hf_hub_token)
    
    # load model from config (for training from scratch)
    logger.info("Training new model from scratch")
    config = AutoConfig.from_pretrained(script_args.model_config_id)
    model = AutoModelForMaskedLM.from_config(config)
    
    logger.info(f"Resizing token embedding to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=script_args.mlm_probability,
        pad_to_multiple_of=8
    )
    
    # define our hyperparameters
    gaudi_training_args = GaudiTrainingArguments(
          output_dir=training_args.output_dir,
          use_habana=True,
          use_lazy_mode=True,
          gaudi_config_name=script_args.gaudi_config_id,
          num_train_epochs=training_args.num_train_epochs,
          per_device_train_batch_size=training_args.per_device_train_batch_size,
          learning_rate=training_args.learning_rate,
          seed=training_args.seed,
          # logging & evaluation strategies
          logging_dir=f"{training_args.output_dir}/logs",
          logging_strategy="steps",
          logging_steps=2000,
          save_strategy="steps",
          save_steps=2000,
          save_total_limit=2,
          report_to="tensorboard",
          # push to hub parameters
          push_to_hub=True,
          hub_strategy="every_save",
          hub_model_id=script_args.repository_id,
          hub_token=script_args.hf_hub_token,
      )

    # Initialize our Trainer
    trainer = GaudiTrainer(
        model=model,
        args=gaudi_training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # train the model
    trainer.train()
    
    
if __name__ == "__main__":
    run_mlm()