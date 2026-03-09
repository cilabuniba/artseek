import json
import os
from functools import partial
from multiprocessing import set_start_method
from pathlib import Path

import click
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from colpali_engine.models import ColQwen2, ColQwen2Processor
from hydra.utils import instantiate
from PIL import Image
from safetensors.torch import load_file, load_model, save_file
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, get_scheduler

from ...data.datasets.artgraph import ArtGraphClassificationDataset
from .metrics import (
    MulticlassClassificationMetrics,
    MultilabelClassificationMetrics,
)
from ...utils.dirutils import get_data_dir
from .li_classification_network import LateInteractionClassificationNetwork

try:
    set_start_method("spawn")
except RuntimeError:
    pass


@click.group
def cli():
    pass


# Collate using the processor
def _collate_fn(batch, data_dir, processor, tasks, precomputed, task_class_counts=None):
    new_batch = {}
    for task in tasks:
        # Get all unique label IDs across the batch for this task
        all_label_ids = [label[0] for item in batch for label in item[task]]
        labels_unique = sorted(set(all_label_ids))

        # Create lists to hold text and one-hot labels for each example
        batch_texts = [None] * len(labels_unique)
        embeddings = [None] * len(labels_unique)
        class_weights = [None] * len(labels_unique)
        batch_one_hot = torch.zeros(len(batch), len(labels_unique))

        # Process each example in the batch
        for i, item in enumerate(batch):
            # Get all labels for this example for the current task
            example_labels = item[task]

            # For each label in this example, set the corresponding bit in the one-hot tensor
            for label_id, label_text in example_labels:
                label_idx = labels_unique.index(label_id)
                batch_texts[label_idx] = label_text
                batch_one_hot[i, label_idx] = 1.0
                if task_class_counts is not None:
                    class_weights[label_idx] = 1.0 / task_class_counts[task][label_id]
                if (
                    precomputed
                    and (
                        data_dir
                        / "precomputed"
                        / task
                        / f"{label_id}.safetensors"
                    ).exists()
                ):
                    tensors = load_file(
                        data_dir
                        / "precomputed"
                        / task
                        / f"{label_id}.safetensors"
                    )
                    embeddings[label_idx] = tensors["embedding"]

        # Process all text labels
        if None not in embeddings:
            queries = {}
        else:
            queries = processor.process_queries(batch_texts)
        # Add one-hot encoded labels to the queries
        queries["labels"] = batch_one_hot
        if class_weights[0] is not None:
            queries["class_weights"] = torch.tensor(class_weights)
        # Add embeddings to the queries if the list does not contain None
        if None not in embeddings:
            # Stack the embeddings into a single tensor, but before that, we need to pad them and create a mask
            max_len = max([embedding.size(0) for embedding in embeddings])
            attention_mask = torch.tensor(
                [
                    [1] * embedding.size(0) + [0] * (max_len - embedding.size(0))
                    for embedding in embeddings
                ],
                dtype=torch.bool,
            )
            embeddings = [
                torch.cat(
                    [
                        embedding,
                        torch.zeros(max_len - embedding.size(0), embedding.size(1)),
                    ]
                )
                for embedding in embeddings
            ]
            embeddings = torch.stack(embeddings)
            queries["embeddings"] = embeddings
            queries["embeddings_mask"] = attention_mask

        # Add example indices to track which text belongs to which example
        example_indices = torch.tensor(
            [i for i, item in enumerate(batch) for _ in item[task]]
        )
        queries["example_indices"] = example_indices

        new_batch[task] = queries

    if "image" in batch[0]:
        images = processor.process_images([item["image"] for item in batch])
        new_batch["image"] = images
    else:
        new_batch["image"] = {}

    # Might read the embeddings from precomputed files
    embeddings = []
    for item in batch:
        if (
            precomputed
            and (
                data_dir
                / "precomputed"
                / "images"
                / f"{item['artwork'][0][1]}.safetensors"
            ).exists()
        ):
            tensors = load_file(
                data_dir
                / "precomputed"
                / "images"
                / f"{item['artwork'][0][1]}.safetensors"
            )
            embeddings.append(tensors["embedding"])
    if len(embeddings) > 0:
        # Stack the embeddings into a single tensor, but before that, we need to pad them and create a mask
        max_len = max([embedding.size(0) for embedding in embeddings])
        attention_mask = torch.tensor(
            [
                [1] * embedding.size(0) + [0] * (max_len - embedding.size(0))
                for embedding in embeddings
            ],
            dtype=torch.bool,
        )
        embeddings = [
            torch.cat(
                [embedding, torch.zeros(max_len - embedding.size(0), embedding.size(1))]
            )
            for embedding in embeddings
        ]
        embeddings = torch.stack(embeddings)
        new_batch["image"]["embeddings"] = embeddings
        new_batch["image"]["embeddings_mask"] = attention_mask

    new_batch["id"] = [item["artwork"][0][0] for item in batch]
    return new_batch


@cli.command
@click.option("--config-path", type=click.Path(exists=True), required=True)
@torch.no_grad()
def precompute(config_path):
    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = instantiate(config_dict)

    train_data_dir = config.datasets.train.data_dir
    val_data_dir = config.datasets.val.data_dir
    test_data_dir = config.datasets.test.data_dir
    tasks = config.tasks

    # Model
    model = ColQwen2.from_pretrained(**config.model)
    processor = ColQwen2Processor.from_pretrained(
        config.model.pretrained_model_name_or_path
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Read data from train.json
    with open(train_data_dir / "train.json", "r") as f:
        data = json.load(f)
    # Also read val and test data, and extend the images key of data
    with open(val_data_dir / "val.json", "r") as f:
        data["artwork"].update(json.load(f)["artwork"])
    with open(test_data_dir / "test.json", "r") as f:
        data["artwork"].update(json.load(f)["artwork"])

    # Create a directory name precomputed in the data directory
    precomputed_dir = train_data_dir / "precomputed"
    precomputed_dir.mkdir(exist_ok=True)

    # Create subdirectories for images and each task
    (precomputed_dir / "images").mkdir(exist_ok=True)
    for task in tasks:
        (precomputed_dir / task).mkdir(exist_ok=True)

    # Precompute image embeddings
    batch_size = 32
    # data["images"] from dict to list
    image_names = [item for item in data["artwork"].values()]
    for i in tqdm(range(0, len(data["artwork"]), batch_size)):
        batch_image_names = image_names[i : i + batch_size]

        # keep only the image_names not already precomputed
        batch_image_names = [
            image_name
            for image_name in batch_image_names
            if not (precomputed_dir / "images" / f"{image_name}.safetensors").exists()
        ]
        if len(batch_image_names) == 0:
            continue

        batch_images = [
            Image.open(train_data_dir / "images" / image_name)
            for image_name in batch_image_names
        ]
        batch_images = processor.process_images(batch_images).to(device)
        embeddings = model(**batch_images)
        for i, (image_name, embedding) in enumerate(zip(batch_image_names, embeddings)):
            # keep only the embeddings of the embedding that are not padding
            embedding = embedding[batch_images.attention_mask[i].to(torch.bool)]
            tensors = {
                "embedding": embedding,
            }
            save_file(tensors, precomputed_dir / "images" / f"{image_name}.safetensors")

    # Precompute text embeddings
    batch_size = 64
    for task in tasks:
        task_list = [item for item in data[task].items()]
        for i in tqdm(range(0, len(task_list), batch_size)):
            batch = task_list[i : i + batch_size]
            batch_ids = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            batch_texts = processor.process_queries(batch_texts).to(device)
            embeddings = model(**batch_texts)
            for i, (label_id, embedding) in enumerate(zip(batch_ids, embeddings)):
                # keep only the embeddings of the embedding that are not padding
                embedding = embedding[batch_texts.attention_mask[i].to(torch.bool)]
                tensors = {
                    "embedding": embedding,
                }
                save_file(tensors, precomputed_dir / task / f"{label_id}.safetensors")


@cli.command
@click.option("--config-path", type=click.Path(exists=True), required=True)
def train(config_path):
    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = instantiate(config_dict)

    # Accelerator
    accelerator = Accelerator(**config.accelerator)

    # Set tracker
    accelerator.init_trackers(
        "ArtSeek",
        config=config_dict,
        init_kwargs={"wandb": {"group": "multimodal_retriever"}},
    )
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    wandb_tracker.define_metric("train/step")
    wandb_tracker.define_metric("val/step")
    wandb_tracker.define_metric("train/*", step_metric="train/step")
    wandb_tracker.define_metric("val/*", step_metric="val/step")

    # Tasks
    tasks = config.tasks

    # Define if embeddings are precomputed
    precomputed = config.training.precomputed

    # Set seed
    seed = config.training.seed
    set_seed(seed)

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not precomputed:
        model = ColQwen2.from_pretrained(**config.model)
    else:
        model = None
    processor = ColQwen2Processor.from_pretrained(
        config.model.pretrained_model_name_or_path
    )
    li_classification_network = LateInteractionClassificationNetwork(
        **config.li_classification_network
    )

    # Loss
    loss = config.loss

    # Optimizer
    optimizer = config.optimizer_partial(
        params=list(li_classification_network.parameters()) + list(loss.parameters())
    )

    # Data
    train_dataset = config.datasets.train
    val_dataset = config.datasets.val
    
    train_data_dir = config.datasets.train.data_dir
    val_data_dir = config.datasets.val.data_dir
    test_data_dir = config.datasets.test.data_dir

    collate_fn_train = partial(
        _collate_fn,
        data_dir=train_data_dir,
        processor=processor,
        tasks=tasks,
        precomputed=precomputed,
        task_class_counts=train_dataset.task_class_counts,
    )
    collate_fn_val = partial(
        _collate_fn,
        data_dir=val_data_dir,
        processor=processor,
        tasks=tasks,
        precomputed=precomputed,
    )

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn_train,
        **config.dataloaders.train,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_fn_val,
        **config.dataloaders.val,
    )

    # Training setup
    class_weighting = config.training.class_weighting
    loss_weighting = config.training.loss_weighting
    max_grad_norm = config.training.max_grad_norm
    num_epochs = config.training.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps), desc="Training Progress")

    # Learning rate scheduler
    lr_scheduler_kwargs = config.lr_scheduler.copy()
    warmup_ratio = lr_scheduler_kwargs.pop("warmup_ratio", None)
    if warmup_ratio is not None:
        num_warmup_steps = int(
            warmup_ratio * len(train_dataloader)
        )  # Warmup steps in the first epoch
    else:
        num_warmup_steps = lr_scheduler_kwargs.pop("num_warmup_steps", 0)

    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        **lr_scheduler_kwargs,
    )

    # Accelerate!
    (
        train_dataloader,
        val_dataloader,
        li_classification_network,
        loss,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        train_dataloader,
        val_dataloader,
        li_classification_network,
        loss,
        optimizer,
        lr_scheduler,
    )

    # Initialize step counters
    train_step = 0
    val_step = 0

    # Early stopping parameters
    early_stopping_patience = config.training.early_stopping_patience
    best_validate_loss = float("inf")
    patience_counter = 0

    # Validation before training
    # validate_loss = validate(
    #     config,
    #     accelerator,
    #     model,
    #     li_classification_network,
    #     loss,
    #     val_dataloader,
    #     val_step,
    #     epoch=0,  # Set epoch to 0 for pre-training validation
    #     progress_bar=progress_bar,  # Pass progress bar to validation
    # )
    # print(f"Validation loss before training: {validate_loss}")

    # Training loop
    for epoch in range(num_epochs):
        li_classification_network.train()
        loss.train()
        epoch_task_losses = {task: [] for task in tasks}
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in train_dataloader:
            task_losses = []
            image_embeddings, image_embeddings_mask = (
                batch["image"]["embeddings"],
                batch["image"]["embeddings_mask"],
            )
            image_task_embeddings = li_classification_network(
                visual_embeddings=image_embeddings,
                visual_mask=image_embeddings_mask,
            )["visual_task_embeddings"]

            for task_idx, task in enumerate(tasks):
                text_embeddings, text_embeddings_mask = (
                    batch[task]["embeddings"],
                    batch[task]["embeddings_mask"],
                )
                text_task_embeddings = li_classification_network(
                    text_embeddings=text_embeddings,
                    text_mask=text_embeddings_mask,
                    task_idx=tasks.index(task),
                )["text_task_embeddings"]
                l = loss(
                    image_task_embeddings[:, task_idx],
                    text_task_embeddings,
                    batch[task]["labels"],
                    txt_weights=batch[task].get("class_weights", None) if class_weighting else None,
                )
                task_losses.append(l)
                epoch_task_losses[task].append(l.item())

            # Compute the combined loss (multi-task learning)
            if not loss_weighting:
                # old implementation
                task_losses_norm = [l / l.detach() for l in task_losses]
                combined_loss = sum(task_losses_norm)
            else:
                # new implementation https://github.com/Mikoto10032/AutomaticWeightedLoss
                combined_loss = 0
                for i, l in enumerate(task_losses):
                    combined_loss += 0.5 / (loss.loss_w[i] ** 2) * l + torch.log(1 + loss.loss_w[i] ** 2)

            # Log task-specific losses and step
            log_dict = {
                f"train/{task}_loss": l.item() for task, l in zip(tasks, task_losses)
            }
            log_dict["train/step"] = train_step
            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
            accelerator.log(log_dict)

            progress_bar.set_postfix(log_dict)

            # Training step
            accelerator.backward(combined_loss)
            accelerator.clip_grad_norm_(list(li_classification_network.parameters()) + list(loss.parameters()), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            train_step += 1

        # Log average task losses for the epoch
        avg_task_losses = {
            f"train/{task}_avg_loss": sum(epoch_task_losses[task])
            / len(epoch_task_losses[task])
            for task in tasks
        }
        avg_task_losses["train/epoch"] = epoch
        avg_task_losses["train/step"] = train_step - 1
        accelerator.log(avg_task_losses)

        # Validation
        validate_loss, val_step = validate(
            config,
            accelerator,
            model,
            li_classification_network,
            loss,
            val_dataloader,
            val_step,
            epoch,
            progress_bar,  # Pass progress bar to validation
        )
        print(f"Validation loss: {validate_loss}")

        # Save checkpoint
        checkpoint_dir = os.path.join(
            accelerator.project_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}"
        )
        accelerator.save_state(output_dir=checkpoint_dir)

        # Early stopping logic
        if validate_loss < best_validate_loss:
            best_validate_loss = validate_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered. No improvement in validation loss for {early_stopping_patience} epochs."
                )
                break


def validate(
    config,
    accelerator,
    model,
    li_classification_network,
    loss,
    dataloader,
    val_step,
    epoch,
    progress_bar=None,  # Optional progress bar
):
    # Set models to eval mode
    if model is not None:
        model.eval()
    li_classification_network.eval()
    loss.eval()

    # Initialize per-epoch accumulators (for each task)
    epoch_losses = {task: [] for task in config.tasks}
    epoch_metrics = {
        task: {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for task in config.tasks
    }

    with torch.no_grad():
        for batch in dataloader:
            # Initialize per-batch dictionaries for logging
            batch_losses = {}
            batch_metrics = {}

            image_embeddings = batch["image"]["embeddings"]
            image_embeddings_mask = batch["image"]["embeddings_mask"]
            image_task_embeddings = li_classification_network(
                visual_embeddings=image_embeddings,
                visual_mask=image_embeddings_mask,
            )["visual_task_embeddings"]

            for task_idx, task in enumerate(config.tasks):
                text_embeddings = batch[task]["embeddings"]
                text_embeddings_mask = batch[task]["embeddings_mask"]
                text_task_embeddings = li_classification_network(
                    text_embeddings=text_embeddings,
                    text_mask=text_embeddings_mask,
                    task_idx=config.tasks.index(task),
                )["text_task_embeddings"]

                # Compute loss and logits for current task
                l, logits = loss(
                    image_task_embeddings[:, task_idx],
                    text_task_embeddings,
                    batch[task]["labels"],
                    return_logits=True,
                )
                curr_loss = l.item()

                # Save batch loss and update epoch loss list
                batch_losses[task] = curr_loss
                epoch_losses[task].append(curr_loss)

                # Compute metrics for current task
                labels = batch[task]["labels"].cpu().numpy()
                logits_np = logits.cpu().numpy()
                task_type = config.task_types[task]

                if task_type == "classification":
                    preds = logits_np.argmax(axis=1)
                    true_labels = labels.argmax(axis=1)
                elif task_type == "multi-label":
                    preds = []
                    true_labels = []
                    for i in range(labels.shape[0]):
                        n = int(labels[i].sum())
                        top_n_indices = logits_np[i].argsort()[-n:][::-1]
                        preds_row = [
                            1 if idx in top_n_indices else 0
                            for idx in range(logits_np.shape[1])
                        ]
                        preds.append(preds_row)
                        true_labels.append(labels[i])
                    preds = torch.tensor(preds)
                    true_labels = torch.tensor(true_labels)

                acc = accuracy_score(true_labels, preds)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    true_labels, preds, average="weighted", zero_division=0
                )

                # Update batch metrics dictionary
                batch_metrics[task] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
                # Store batch metrics into epoch accumulators
                epoch_metrics[task]["accuracy"].append(acc)
                epoch_metrics[task]["precision"].append(prec)
                epoch_metrics[task]["recall"].append(rec)
                epoch_metrics[task]["f1"].append(f1)

            # Log per-batch losses and metrics
            batch_log_dict = {}
            for task in config.tasks:
                batch_log_dict[f"val/{task}_loss"] = batch_losses[task]
                batch_log_dict[f"val/{task}_accuracy"] = batch_metrics[task]["accuracy"]
                batch_log_dict[f"val/{task}_precision"] = batch_metrics[task][
                    "precision"
                ]
                batch_log_dict[f"val/{task}_recall"] = batch_metrics[task]["recall"]
                batch_log_dict[f"val/{task}_f1"] = batch_metrics[task]["f1"]
            batch_log_dict["val/step"] = val_step
            accelerator.log(batch_log_dict)

            # Update the progress bar with batch-level stats if provided
            if progress_bar:
                progress_bar.set_postfix(batch_log_dict)
            val_step += 1

    # After all batches, log epoch-level average metrics for each task
    epoch_log_dict = {}
    for task in config.tasks:
        epoch_log_dict[f"val/{task}_avg_loss"] = sum(epoch_losses[task]) / len(
            epoch_losses[task]
        )
        epoch_log_dict[f"val/{task}_avg_accuracy"] = sum(
            epoch_metrics[task]["accuracy"]
        ) / len(epoch_metrics[task]["accuracy"])
        epoch_log_dict[f"val/{task}_avg_precision"] = sum(
            epoch_metrics[task]["precision"]
        ) / len(epoch_metrics[task]["precision"])
        epoch_log_dict[f"val/{task}_avg_recall"] = sum(
            epoch_metrics[task]["recall"]
        ) / len(epoch_metrics[task]["recall"])
        epoch_log_dict[f"val/{task}_avg_f1"] = sum(epoch_metrics[task]["f1"]) / len(
            epoch_metrics[task]["f1"]
        )

    epoch_log_dict["val/epoch"] = epoch
    epoch_log_dict["val/step"] = val_step - 1
    accelerator.log(epoch_log_dict)

    # Update the progress bar with epoch-level stats if provided
    if progress_bar:
        progress_bar.set_postfix(epoch_log_dict)

    # Compute and return the overall average loss across tasks
    avg_loss_across_tasks = sum(
        [epoch_log_dict[f"val/{task}_avg_loss"] for task in config.tasks]
    ) / len(config.tasks)
    return avg_loss_across_tasks, val_step


@cli.command
@click.option("--config-path", type=click.Path(exists=True), required=True)
@torch.no_grad()
def test(config_path):
    # Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = instantiate(config_dict)

    tasks = config.tasks
    task_types = config.task_types

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    li_classification_network = LateInteractionClassificationNetwork(
        **config.li_classification_network
    ).to(device)
    checkpoint_dir = os.path.join(config.accelerator.project_dir, "checkpoints")
    # find the last checkpoint
    checkpoints = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")
    ]
    checkpoints_lookup = {
        int(checkpoint.split("_")[-1]): checkpoint for checkpoint in checkpoints
    }
    last_checkpoint = checkpoints_lookup[max(checkpoints_lookup.keys())]
    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint, "model.safetensors")
    load_model(li_classification_network, checkpoint_path)
    li_classification_network.eval()

    # Load loss
    loss = config.loss.to(device)
    loss_checkpoint_path = os.path.join(
        checkpoint_dir, last_checkpoint, "model_1.safetensors"
    )
    load_model(loss, loss_checkpoint_path)
    loss.eval()

    # Define the task matrices for each task and fill them
    data_dir = config.datasets.test.data_dir
    # Read valid labels
    # with open(os.path.join(data_dir, "valid_labels.json"), "r") as f:
    #     valid_labels = json.load(f)
    precomputed_dir = os.path.join(data_dir, "precomputed")
    task_matrices = {}
    task_lookups = {}
    batch_size = 64
    for task_idx, task in enumerate(tasks):
        task_dir = os.path.join(precomputed_dir, task)
        task_lookups[task] = {"-100": -100}
        # count the safetensors files
        filenames = [f for f in os.listdir(task_dir) if f.endswith(".safetensors")]
        n_files = len(filenames)
        # n_classes = len(valid_labels[task])
        n_classes = n_files
        task_matrix = torch.zeros(
            n_classes, config.li_classification_network.output_dim
        ).to(device)

        # load batches of embeddings, process them with the model and fill the task matrix
        real_i = 0
        for i in tqdm(range(0, n_files, batch_size)):
            batch = []
            real_batch_size = 0
            for j, filename in enumerate(filenames[i : i + batch_size]):
                class_id = filename.removesuffix(".safetensors")
                # if not int(class_id) in valid_labels[task]:
                #     continue
                tensors = load_file(os.path.join(task_dir, filename))
                batch.append(tensors["embedding"])
                task_lookups[task][class_id] = real_batch_size + real_i
                real_batch_size += 1
                
            if real_batch_size == 0:
                continue

            # find the max length of the embeddings
            max_len = max([embedding.size(0) for embedding in batch])
            # create the mask
            attention_mask = torch.tensor(
                [
                    [1] * embedding.size(0) + [0] * (max_len - embedding.size(0))
                    for embedding in batch
                ],
                dtype=torch.bool,
            ).to(device)
            # pad the embeddings
            batch = [
                torch.cat(
                    [
                        embedding,
                        torch.zeros(max_len - embedding.size(0), embedding.size(1)),
                    ]
                )
                for embedding in batch
            ]
            batch = torch.stack(batch).to(device)

            # process the batch with the model
            task_matrix[real_i : real_i + real_batch_size] = li_classification_network(
                text_embeddings=batch, text_mask=attention_mask, task_idx=task_idx
            )["text_task_embeddings"]
            real_i += real_batch_size
        task_matrices[task] = task_matrix

    # Save task lookups
    with open(os.path.join(config.accelerator.project_dir, "task_lookups.json"), "w") as f:
        json.dump(task_lookups, f, indent=4)
    # Save task matrices as safetensors
    save_file(task_matrices, os.path.join(config.accelerator.project_dir, "task_matrices.safetensors"))

    # Load the test dataset
    test_dataset = config.datasets.test
    test_dataloder = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=partial(
            _collate_fn,
            processor=ColQwen2Processor.from_pretrained(
                config.model.pretrained_model_name_or_path
            ),
            tasks=tasks,
            precomputed=config.training.precomputed,
        ),
        **config.dataloaders.test,
    )

    # Setup metrics
    metrics = {}
    for task in tasks:
        if task_types[task] == "classification":
            num_classes = task_matrices[task].size(0)
            metrics[task] = MulticlassClassificationMetrics(num_classes).to(device)
        else:
            num_labels = task_matrices[task].size(0)
            metrics[task] = MultilabelClassificationMetrics(num_labels).to(device)

    # Evaluate the model on the test dataset
    for batch in tqdm(test_dataloder):
        # Process the batch with the model
        image_embeddings = batch["image"]["embeddings"].to(device)
        image_embeddings_mask = batch["image"]["embeddings_mask"].to(device)
        image_task_embeddings = li_classification_network(
            visual_embeddings=image_embeddings,
            visual_mask=image_embeddings_mask,
        )["visual_task_embeddings"]

        # Load the labels
        labels = {}
        for task_idx, task in enumerate(tasks):
            labels[task] = []
            for id in batch["id"]:
                relationships = test_dataset.annotations["relationships"][id]
                if task in relationships:
                    relationship_labels = test_dataset.annotations["relationships"][id][task]
                    # keep only the labels that are in the valid labels
                    relationship_labels = [
                        label for label in relationship_labels if label in valid_labels[task]
                    ]
                    labels[task].append(
                        relationship_labels
                    )
                else:
                    labels[task].append([])
                if task_types[task] == "classification":
                    if len(labels[task][-1]) == 0:
                        labels[task][-1].append(-100)
                    labels[task][-1] = task_lookups[task][str(labels[task][-1][0])]
                else:
                    one_hot_list = [0] * len(task_matrices[task])
                    for label in labels[task][-1]:
                        one_hot_list[task_lookups[task][str(label)]] = 1
                    labels[task][-1] = one_hot_list
            labels[task] = torch.tensor(labels[task]).to(device)

            # Get the logits for each task
            logits = loss(
                image_task_embeddings[:, task_idx],
                task_matrices[task],
                None,
                return_logits=True,
            )
            # Apply sigmoid to the logits
            preds = torch.sigmoid(logits)

            # If the task is multi-label and a labels row is all zeros, ignore it
            if task_types[task] == "multi-label":
                # Find the indices of the rows that are all zeros
                valid_rows = labels[task].sum(dim=1) > 0
                preds = preds[valid_rows]
                labels[task] = labels[task][valid_rows]
            
            # Update the metrics
            metrics[task].update(preds, labels[task])

    # Save the metrics to a json file
    metrics_dict = {}
    for task in tasks:
        metrics_dict[task] = metrics[task].compute()
        for metric in metrics_dict[task]:
            metrics_dict[task][metric] = metrics_dict[task][metric].item()
    with open(os.path.join(config.accelerator.project_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
        
@cli.command(name="export-task-matrices")
@click.option("--config-path", type=click.Path(exists=True), required=True, help="Path to the model config file.")
@click.option("--output-dir", type=click.Path(), required=True, help="Directory path to save lookups and matrices.")
@torch.no_grad()
def export_task_matrices(config_path, output_dir):
    """Computes and saves task lookups and matrices to a specified directory."""
    
    # 1. Load config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = instantiate(config_dict)

    tasks = config.tasks

    # 2. Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    li_classification_network = LateInteractionClassificationNetwork(
        **config.li_classification_network
    ).to(device)
    
    checkpoint_dir = os.path.join(config.accelerator.project_dir, "checkpoints")
    # find the last checkpoint
    checkpoints = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")
    ]
    checkpoints_lookup = {
        int(checkpoint.split("_")[-1]): checkpoint for checkpoint in checkpoints
    }
    last_checkpoint = checkpoints_lookup[max(checkpoints_lookup.keys())]
    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint, "model.safetensors")
    
    load_model(li_classification_network, checkpoint_path)
    li_classification_network.eval()

    # 3. Define and fill the task matrices
    data_dir = config.datasets.test.data_dir
    precomputed_dir = os.path.join(data_dir, "precomputed")
    
    task_matrices = {}
    task_lookups = {}
    batch_size = 64
    
    for task_idx, task in enumerate(tasks):
        task_dir = os.path.join(precomputed_dir, task)
        task_lookups[task] = {"-100": -100}
        
        # count the safetensors files
        filenames = [f for f in os.listdir(task_dir) if f.endswith(".safetensors")]
        n_files = len(filenames)
        n_classes = n_files
        
        task_matrix = torch.zeros(
            n_classes, config.li_classification_network.output_dim
        ).to(device)

        # load batches of embeddings, process them with the model and fill the task matrix
        real_i = 0
        for i in tqdm(range(0, n_files, batch_size), desc=f"Processing {task}"):
            batch = []
            real_batch_size = 0
            for j, filename in enumerate(filenames[i : i + batch_size]):
                class_id = filename.removesuffix(".safetensors")
                tensors = load_file(os.path.join(task_dir, filename))
                batch.append(tensors["embedding"])
                task_lookups[task][class_id] = real_batch_size + real_i
                real_batch_size += 1
                
            if real_batch_size == 0:
                continue

            # find the max length of the embeddings
            max_len = max([embedding.size(0) for embedding in batch])
            # create the mask
            attention_mask = torch.tensor(
                [
                    [1] * embedding.size(0) + [0] * (max_len - embedding.size(0))
                    for embedding in batch
                ],
                dtype=torch.bool,
            ).to(device)
            # pad the embeddings
            batch_padded = [
                torch.cat(
                    [
                        embedding,
                        torch.zeros(max_len - embedding.size(0), embedding.size(1)),
                    ]
                )
                for embedding in batch
            ]
            batch_tensor = torch.stack(batch_padded).to(device)

            # process the batch with the model
            task_matrix[real_i : real_i + real_batch_size] = li_classification_network(
                text_embeddings=batch_tensor, text_mask=attention_mask, task_idx=task_idx
            )["text_task_embeddings"]
            real_i += real_batch_size
            
        task_matrices[task] = task_matrix

    # 4. Save to the provided CLI output directory
    os.makedirs(output_dir, exist_ok=True)
    
    lookups_path = os.path.join(output_dir, "task_lookups.json")
    with open(lookups_path, "w") as f:
        json.dump(task_lookups, f, indent=4)
        
    matrices_path = os.path.join(output_dir, "task_matrices.safetensors")
    save_file(task_matrices, matrices_path)
    
    click.echo(f"Successfully saved task lookups and matrices to: {output_dir}")


if __name__ == "__main__":
    cli()
