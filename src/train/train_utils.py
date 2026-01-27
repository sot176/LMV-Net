import torch
import numpy as np
from accelerate import Accelerator
from utils.c_index import concordance_index_ipcw, get_censoring_dist
from utils.utils import get_risk_loss_BCE, compute_auc_x_year_auc


def train_one_epoch(args, model_risk, train_loader, optimizer, accelerator,  warmup_scheduler, global_step, warmup_steps):
    """
    Handles the training logic for a single epoch.

    Returns:
        A tuple containing (average_risk_loss, c_index) for the epoch.
    """
    model_risk.train()
    running_risk_loss = 0.0
    all_preds, all_times, all_events = [], [], []

    for batch in train_loader:

        outputs = model_risk(batch["current_image_cc"], batch["previous_image_cc"],batch["current_image_mlo"], batch["previous_image_mlo"],
                             batch["time_gap"])

        if args.use_only_multiview_pred:
            risk_multi = outputs["risk_multi"]
            risk_loss_multi = get_risk_loss_BCE(risk_multi, batch["target"], batch["y_mask"])
            risk_loss = risk_loss_multi
        else:
            risk_multi = outputs["risk_multi"]
            risk_cc = outputs["risk_cc"]
            risk_mlo = outputs["risk_mlo"]

            risk_loss_multi = get_risk_loss_BCE(risk_multi, batch["target"], batch["y_mask"])
            risk_loss_cc = get_risk_loss_BCE(risk_cc, batch["target"], batch["y_mask"])
            risk_loss_mlo = get_risk_loss_BCE(risk_mlo, batch["target"], batch["y_mask"])

            risk_loss = risk_loss_multi + risk_loss_cc + risk_loss_mlo

        running_risk_loss += risk_loss.item()

        optimizer.zero_grad()
        accelerator.backward(risk_loss)
        optimizer.step()

        # Update warm-up scheduler
        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

        # Gather results for metric calculation
        all_preds.append(accelerator.gather(torch.sigmoid(risk_multi).detach()))
        all_times.append(accelerator.gather(batch["event_times"]))
        all_events.append(accelerator.gather(batch["event_observed"]))

    avg_risk_loss = running_risk_loss / len(train_loader)
    c_index, auc_results = 0, {}
    # Calculate metrics on the main process
    if accelerator.is_main_process:
        preds = torch.cat(all_preds).cpu().numpy()
        times = torch.cat(all_times).cpu().numpy().astype(int)
        events = torch.cat(all_events).cpu().numpy()
        censoring_dist = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring_dist)
        auc_results = compute_auc_x_year_auc(preds, times, events)

    return avg_risk_loss, c_index, auc_results


def evaluate(args, model_risk, valid_loader, accelerator):
    """
    Handles the evaluation logic for a single epoch.

    Returns:
        A tuple containing (average_risk_loss, c_index, auc_results).
    """
    model_risk.eval()
    running_risk_loss = 0.0
    val_preds, val_times, val_events = [], [], []

    with torch.no_grad():
        for batch_val in valid_loader:
            outputs_val = model_risk(batch_val["current_image_cc"], batch_val["previous_image_cc"], batch_val["current_image_mlo"],
                                 batch_val["previous_image_mlo"],
                                 batch_val["time_gap"])
            if args.use_only_multiview_pred:
                risk_pred_val = outputs_val["risk_multi"]
                risk_loss_multi = get_risk_loss_BCE(risk_pred_val, batch_val["target"], batch_val["y_mask"])
                risk_loss_val = risk_loss_multi
            else:
                risk_pred_val = outputs_val["risk_multi"]
                risk_cc = outputs_val["risk_cc"]
                risk_mlo = outputs_val["risk_mlo"]

                risk_loss_multi = get_risk_loss_BCE(risk_pred_val, batch_val["target"], batch_val["y_mask"])
                risk_loss_cc = get_risk_loss_BCE(risk_cc, batch_val["target"], batch_val["y_mask"])
                risk_loss_mlo = get_risk_loss_BCE(risk_mlo, batch_val["target"], batch_val["y_mask"])

                risk_loss_val = risk_loss_multi + risk_loss_cc + risk_loss_mlo
            running_risk_loss += risk_loss_val.item()

            val_preds.append(accelerator.gather(torch.sigmoid(risk_pred_val).detach()))
            val_times.append(accelerator.gather(batch_val["event_times"]))
            val_events.append(accelerator.gather(batch_val["event_observed"]))

    avg_risk_loss = running_risk_loss / len(valid_loader)
    c_index, auc_results = 0, {}
    # Calculate metrics on the main process
    if accelerator.is_main_process:
        predictions_val = torch.cat(val_preds).cpu().numpy()
        event_times_val = torch.cat(val_times).cpu().numpy().astype(int)
        event_observed_val = torch.cat(val_events).cpu().numpy()
        censoring_dist_val = get_censoring_dist(event_times_val, event_observed_val)
        c_index = concordance_index_ipcw(event_times_val, predictions_val, event_observed_val, censoring_dist_val)
        auc_results = compute_auc_x_year_auc(predictions_val, event_times_val, event_observed_val)

    return avg_risk_loss, c_index, auc_results
