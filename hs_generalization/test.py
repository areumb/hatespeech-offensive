import functools
import json
import logging
from pathlib import Path
from typing import Tuple, Any, Dict

import click
import evaluate
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Metric
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification

from hs_generalization.train import get_dataloader, combine_compute
from hs_generalization.utils import load_config, get_dataset

import click #AK
from hs_generalization.utils import load_config #AK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


def evaluate_data(
        model: Any,
        dataloader: DataLoader,
        metric: Metric,
        device: str,
) -> Tuple[np.float_, Dict, Any, Any, Any, Any]:
    """Function for running model on evaluation or test set.
    In this function, the model loaded from checkpoint is applied on the evaluation or test set from a dataloader.
    Loss and accuracy are tracked as well.
    Args:
        model (Model): Model that is being trained.:
        dataloader (DataLoader): Object that will load the training data.
        metric (Metric): Metric that is being tracked.
        device (str): Device on which training will be done.
    Returns:
        eval_loss (float): Average loss over the whole validation set.
        eval_accuracy (float): Average accuracy over the whole validation set.
    """
    model.eval()

    predictions = torch.tensor([])
    references = torch.tensor([])
    confidences = torch.tensor([])

    with torch.no_grad():
        losses = []
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            references = torch.cat([references, batch["labels"].to("cpu")])
            confidences = torch.cat([confidences, outputs.logits.softmax(dim=-1).to("cpu")])

            losses.append(outputs.loss.detach().cpu().numpy())

    eval_loss = np.mean(losses)
    score_micro = metric.compute(predictions=predictions, references=references, average="micro")
    score_macro = metric.compute(predictions=predictions, references=references, average="macro")
    metrics_micro = {f"eval_micro_{name}": val for name, val in score_micro.items()}
    metrics_macro = {f"eval_macro_{name}": val for name, val in score_macro.items()}
    metrics = metrics_micro | metrics_macro

    cm = confusion_matrix(references, predictions)
    return eval_loss, metrics, predictions.int().tolist(), cm, confidences.tolist(), references.int().tolist()



@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
@click.option("--seed", type=int, default=None, help="Seed number to update config paths") #AK
@click.option("--checkpoint", type=str, default=None, help="Override checkpoint path") #AK

def main(config_path: str, seed: int, checkpoint: str): #AK
    """Function that executes the entire training pipeline.
    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.
    Args:
        config_path (str): path to the config file for the training experiment.
    """
    config = load_config(config_path)

    #AK: If the --seed argument is provided, update checkpoint and output paths.
    if seed is not None:
        config["pipeline"]["seed"] = int(config["pipeline"]["seed"].format(seed=seed))
        config["task"]["checkpoint"] = config["task"]["checkpoint"].format(seed=seed)
        config["pipeline"]["output_predictions"] = config["pipeline"]["output_predictions"].format(seed=seed)
        config["wandb"]["run_name"] = config["wandb"]["run_name"].format(seed=seed)

    # AK: Override checkpoint if provided 
    if checkpoint is not None:
        config["task"]["checkpoint"] = checkpoint


    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    wandb.init(config=config, project=config["wandb"]["project_name"], name=config["wandb"]["run_name"],
               mode="offline"  #AK This forces WandB to run in offline mode; added it to stop typing 3 ("don't visualize")
               )

    accelerator = Accelerator()
    device = accelerator.device

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    dataset_directory = config["task"].get("dataset_directory")
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device", device)
    padding = config["processing"]["padding"]

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        dataset_name,
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
        dataset_directory=dataset_directory,
    )
    dataset = dataset["test"]


#-----------------------------------------------------------------------------------------------------------------
    # for 3classes-3classes: let it be without edits from the original



    #--HATE vs NON-HATE EVAL: take all examples, but merge 1+2 -> non-hate(1) ---
   # def merge_to_non_hate(example):
        # Davidson: 0=hate, 1=offensive, 2=clean
  #      example["labels"] = 0 if example["labels"] == 0 else 1
    #    return example
   # dataset = dataset.map(merge_to_non_hate)


    #--Not Clean vs Clean EVAL: take all examples, but merge 0+1 -> non-clean (0), clean(2) becomes 1 ---
    def merge_to_non_clean(example):
        # Davidson: 0=hate, 1=offensive, 2=clean
        example["labels"] = 1 if example["labels"] == 2 else 0
        return example
    dataset = dataset.map(merge_to_non_clean)




    #-- For Hate vs Clean testing on Hate vs Clean----------------
    # PURE BINARY EVAL FOR Hate vs Clean tested on Hate vs Clean: DROP offensive (orig label 1)
   # dataset = dataset.filter(lambda ex: ex["labels"] != 1) #remove offensive
    # then REMAP the remaining labels: hate 0->hate(0), clean 2->clean(1)
  #  def merge_labels(example):
        # Davidson: 0=hate, 1=offensive, 2=neither
     #   example["labels"] = 0 if example["labels"] == 0 else 1
    #   return example
   # dataset = dataset.map(merge_labels)



    batch_size = config["pipeline"]["batch_size"]
    dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)

    # Load metric, model, optimizer, and learning rate scheduler.
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    metric.compute = functools.partial(combine_compute, metric)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    logger.info(f" Device: {device}.")
    logger.info(" Starting evaluating model on the data.")
    eval_loss, eval_accuracy, predictions, cm, confidences, references = evaluate_data(model, dataloader, metric, device)
    logger.info(f" Average Loss: {eval_loss}, Average Accuracy: {eval_accuracy}")

    save_dict = {
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "average_loss": float(eval_loss),
        "results": eval_accuracy,
        "confidences": confidences,
        "references": references,
    }

    if "output_predictions" in config["pipeline"]:
        p = Path(config["pipeline"]["output_predictions"]).parent
        p.mkdir(exist_ok=True, parents=True)

        with open(config["pipeline"]["output_predictions"], "w") as f:
            json.dump(save_dict, f)

    wandb.log(save_dict)

    wandb.finish()#AK

if __name__ == "__main__":
    main()
