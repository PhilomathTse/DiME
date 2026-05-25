from asyncio.log import logger
import torch
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
from datetime import datetime
from fvcore.nn import parameter_count
import time

from src.common.datasets.mmcsd import load_and_preprocess_data_mmcsd
from src.common.datasets.MultiModalDataset import create_loaders

from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    plot_interaction_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
    visualize_expert_logits_distribution,
    set_style,
)

from src.dime.dime_model import DiME

set_style()


def _move_batch_to_device(batch_samples, device):
    """
    Move batch samples to device.

    Supports both:
    1. Tensor modality input
    2. Dict modality input, e.g. BERT input:
       {
           "input_ids": ...,
           "attention_mask": ...,
           "topic_input_ids": ...,
           "topic_attention_mask": ...
       }
    """
    return {
        k: (
            {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
            if isinstance(v, dict)
            else v.to(device, non_blocking=True)
        )
        for k, v in batch_samples.items()
    }


def _build_fusion_input(
    batch_samples,
    encoder_dict,
    modalities_in_order,
    args,
):
    """
    Encode each modality in args.modality order.

    For MMCSD:
        T = text
        S = screenshot/image

    If args.replace_text_with_topic=True, the text modality replacement tensor
    is generated from topic_input_ids/topic_attention_mask.
    """
    fusion_input = []
    replacement_tensors = {}

    for i, modality in enumerate(modalities_in_order):
        samples = batch_samples[modality]
        enc_out = encoder_dict[modality](samples)
        fusion_input.append(enc_out)

        if (
            modality == "T"
            and isinstance(samples, dict)
            and "topic_input_ids" in samples
            and getattr(args, "replace_text_with_topic", False)
        ):
            topic_inputs = {
                "input_ids": samples["topic_input_ids"],
                "attention_mask": samples["topic_attention_mask"],
            }
            with torch.no_grad():
                topic_vec = encoder_dict[modality](topic_inputs)
            replacement_tensors[i] = topic_vec

    return fusion_input, replacement_tensors


def _safe_multiclass_auc(all_labels, all_probs, n_labels):
    """
    Compute multi-class AUC safely.

    If a validation/test split does not contain all classes, roc_auc_score may fail.
    In that case, return 0.0 instead of crashing.
    """
    try:
        return roc_auc_score(
            np.array(all_labels),
            np.array(all_probs),
            multi_class="ovo",
            labels=list(range(n_labels)),
        )
    except ValueError:
        return 0.0


def train_and_evaluate_dime(args, seed, fusion_model, fusion):
    """
    Train and evaluate Interaction-MoE on MMCSD only.

    Returns:
        (
            best_val_acc,
            best_val_f1,
            best_val_auc,
            test_acc,
            test_f1,
            test_f1_micro,
            test_auc,
            train_time_per_epoch,
            infer_time,
            total_flop,
            total_param,
        )
    """
    seed_everything(seed)

    args.data = "mmcsd"

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)

    num_modalities = len(args.modality)
    modalities_in_order = list(args.modality)

    (
        data_dict,
        encoder_dict,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        n_labels,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        _,
        _,
    ) = load_and_preprocess_data_mmcsd(args)

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.use_common_ids,
        dataset=args.data,
    )

    ensemble_model = DiME(
        num_modalities=num_modalities,
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=args.fusion_sparse,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]

    def log_param_count(params, logger):
        unique_params = list({id(p): p for p in params}.values())

        total_params = sum(p.numel() for p in unique_params)
        trainable_params = sum(p.numel() for p in unique_params if p.requires_grad)

        logger.info(f"Unique params: {len(unique_params)} tensors")
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

        print(f"Unique params: {len(unique_params)} tensors")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

    log_param_count(params, logger)

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_auc = 0.0

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "interaction": [], "gate": []}
    else:
        plotting_total_losses = {"task": [], "interaction": []}

    plotting_interaction_losses = {}
    for i in range(len(args.modality)):
        plotting_interaction_losses[f"uni_{i + 1}"] = []
    plotting_interaction_losses["red"] = []

    patience = getattr(args, "patience", 10)
    no_improve = 0
    best_epoch = -1

    train_time = 0.0

    best_model_fus = deepcopy(ensemble_model.state_dict())
    best_model_enc = {
        modality: deepcopy(encoder.state_dict())
        for modality, encoder in encoder_dict.items()
    }

    if args.save:
        best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
        best_model_enc_cpu = {
            modality: {k: v.cpu() for k, v in enc_state.items()}
            for modality, enc_state in best_model_enc.items()
        }

    for epoch in trange(args.train_epochs):
        epoch_start_time = time.time()

        ensemble_model.train()
        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        batch_interaction_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []

        num_interaction_experts = len(args.modality) + 1
        interaction_loss_sums = [0.0] * num_interaction_experts
        minibatch_count = len(train_loader)

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = _move_batch_to_device(batch_samples, device)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)

            optimizer.zero_grad()

            fusion_input, replacement_tensors = _build_fusion_input(
                batch_samples=batch_samples,
                encoder_dict=encoder_dict,
                modalities_in_order=modalities_in_order,
                args=args,
            )

            if args.fusion_sparse:
                _, _, outputs, interaction_losses, gate_losses = ensemble_model(
                    fusion_input,
                    replacement_tensors=replacement_tensors,
                )
            else:
                _, _, outputs, interaction_losses = ensemble_model(
                    fusion_input,
                    replacement_tensors=replacement_tensors,
                )

            task_loss = criterion(outputs, batch_labels)
            interaction_loss = sum(interaction_losses) / len(args.modality)

            if args.fusion_sparse:
                gate_loss = torch.mean(
                    torch.stack(
                        [
                            g if torch.is_tensor(g) else torch.tensor(g, device=device)
                            for g in gate_losses
                        ]
                    )
                )
                loss = (
                    task_loss
                    + args.interaction_loss_weight * interaction_loss
                    + args.gate_loss_weight * gate_loss
                )
            else:
                loss = task_loss + args.interaction_loss_weight * interaction_loss

            loss.backward()
            optimizer.step()

            batch_task_losses.append(task_loss.item())
            batch_interaction_losses.append(interaction_loss.item())
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())

            for idx, i_loss in enumerate(interaction_losses):
                interaction_loss_sums[idx] += i_loss.item()

        epoch_end_time = time.time()
        train_time += epoch_end_time - epoch_start_time

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        plotting_total_losses["interaction"].append(np.mean(batch_interaction_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))

        for i in range(len(args.modality)):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i + 1}"].append(avg_loss)

        plotting_interaction_losses["red"].append(
            interaction_loss_sums[-1] / minibatch_count
        )

        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

        with torch.no_grad():
            for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                batch_samples = _move_batch_to_device(batch_samples, device)
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_mcs = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)

                fusion_input = []

                for modality in modalities_in_order:
                    samples = batch_samples[modality]
                    enc_out = encoder_dict[modality](samples)
                    fusion_input.append(enc_out)

                _, _, outputs = ensemble_model.inference(fusion_input)

                val_loss = criterion(outputs, batch_labels)
                val_losses.append(val_loss.item())

                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)

                if probs.shape[1] != n_labels:
                    raise ValueError("Incorrect output shape from the model")

        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_auc = _safe_multiclass_auc(all_labels, all_probs, n_labels)

        print(
            f"[Seed {seed}/{args.n_runs - 1}] "
            f"[Epoch {epoch + 1}/{args.train_epochs}] "
            f"Val Loss: {val_loss:.2f}, "
            f"Val Acc: {val_acc * 100:.2f}, "
            f"Val F1: {val_f1 * 100:.2f}, "
            f"Val AUC: {val_auc * 100:.2f}"
        )

        improved = False

        if val_f1 > best_val_f1:
            improved = True
            print(
                f" [(**Best**) Epoch {epoch + 1}/{args.train_epochs}] "
                f"Val Acc: {val_acc * 100:.2f}, "
                f"Val F1: {val_f1 * 100:.2f}, "
                f"Val AUC: {val_auc * 100:.2f}"
            )

            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc

            best_model_fus = deepcopy(ensemble_model.state_dict())
            best_model_enc = {
                modality: deepcopy(encoder.state_dict())
                for modality, encoder in encoder_dict.items()
            }

            if args.save:
                best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                best_model_enc_cpu = {
                    modality: {k: v.cpu() for k, v in enc_state.items()}
                    for modality, enc_state in best_model_enc.items()
                }

        if improved:
            no_improve = 0
            best_epoch = epoch + 1
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[Early Stop] No improvement for {patience} validations. "
                    f"Best Val F1 at epoch {best_epoch}."
                )
                break

    total_param = parameter_count(ensemble_model)[""]
    total_flop = 0

    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="dime",
        fusion=fusion,
    )

    plot_interaction_loss_curves(
        args,
        plotting_interaction_losses=plotting_interaction_losses,
        framework="dime",
        fusion=fusion,
    )

    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/dime/{fusion}/{args.data}").mkdir(exist_ok=True, parents=True)

        save_path = (
            f"./saves/dime/{fusion}/{args.data}/"
            f"{args.target}_seed_{seed}_modality_{args.modality}_"
            f"train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
        )

        torch.save(
            {
                "ensemble_model": best_model_fus_cpu,
                "encoder_dict": best_model_enc_cpu,
            },
            save_path,
        )

        print(f"Best model saved to {save_path}")

    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    all_routing_weights = []

    num_experts = len(args.modality) + 1
    all_expert_outputs = [[] for _ in range(num_experts)]

    infer_time = 0.0

    with torch.no_grad():
        epoch_start_time = time.time()

        for (
            batch_samples,
            batch_ids,
            batch_labels,
            batch_mcs,
            batch_observed,
        ) in test_loader:
            batch_samples = _move_batch_to_device(batch_samples, device)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)

            fusion_input = []

            for modality in modalities_in_order:
                samples = batch_samples[modality]
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            expert_outputs, routing_weights, outputs = ensemble_model.inference(
                fusion_input
            )

            for expert_idx in range(num_experts):
                all_expert_outputs[expert_idx].extend(
                    expert_outputs[expert_idx].cpu().numpy()
                )

            all_routing_weights.extend(routing_weights.cpu().numpy())

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_ids.extend(batch_ids.cpu().numpy())

            all_probs.extend(
                torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            )

        epoch_end_time = time.time()
        infer_time += epoch_end_time - epoch_start_time

    visualize_expert_logits(
        expert_outputs,
        routing_weights,
        outputs,
        args,
        framework="dime",
        fusion=fusion,
    )

    visualize_expert_logits_distribution(
        all_expert_outputs,
        args,
        framework="dime",
        fusion=fusion,
    )

    visualize_sample_weights(
        all_routing_weights,
        args,
        framework="dime",
        fusion=fusion,
    )

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    test_f1_micro = f1_score(all_labels, all_preds, average="micro")
    test_auc = _safe_multiclass_auc(all_labels, all_probs, n_labels)

    now = datetime.now()
    save_dir = Path(
        f"./outputs/dime/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
    np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
    np.save(save_dir / "all_preds.npy", np.array(all_preds))
    np.save(save_dir / "all_labels.npy", np.array(all_labels))
    np.save(save_dir / "all_ids.npy", np.array(all_ids))

    return (
        best_val_acc,
        best_val_f1,
        best_val_auc,
        test_acc,
        test_f1,
        test_f1_micro,
        test_auc,
        train_time / args.train_epochs,
        infer_time,
        total_flop,
        total_param,
    )