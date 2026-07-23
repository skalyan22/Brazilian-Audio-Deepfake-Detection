"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
import numpy as np
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from brspeech_dataset import BRSpeechDataset
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve)

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the BRSpeech dataset model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])

    # define model related paths
    model_tag = "{}_ep{}_bs{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        eval_metrics = evaluate(eval_loader, model, device)
        eval_eer = eval_metrics["eer"]
        eval_auc = eval_metrics["auc"]
        eval_acc = eval_metrics["balanced_acc"]
        
        print("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        metrics = evaluate(dev_loader, model, device)
        dev_eer = metrics["eer"]
        dev_auc = metrics["auc"]
        dev_acc = metrics["balanced_acc"]
        
        print(
            f"Loss={running_loss:.4f} | "
            f"Balanced Acc={dev_acc:.4f} | "
            f"AUC={dev_auc:.4f} | "
            f"EER={dev_eer:.4f}"
        )
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                metrics_eval = evaluate(eval_loader, model, device)
                eval_eer = metrics_eval["eer"]
                eval_auc = metrics_eval["auc"]
                eval_acc = metrics_eval["balanced_acc"]

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    eval_metrics = evaluate(eval_loader, model, device)
    eval_eer = eval_metrics["eer"]
    eval_auc = eval_metrics["auc"]
    eval_acc = eval_metrics["balanced_acc"]
    
    print(
        f"Loss={running_loss:.4f} | "
        f"Balanced Acc={eval_acc:.4f} | "
        f"AUC={eval_auc:.4f} | "
        f"EER={eval_eer:.4f}"
    )
    
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_auc, eval_acc))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
        
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(database_path, seed, config):

    train_set = BRSpeechDataset(
        database_path,
        split=0,
        nb_samp=config["model_config"]["nb_samp"],
    )

    dev_set = BRSpeechDataset(
        database_path,
        split=1,
        nb_samp=config["model_config"]["nb_samp"],
    )

    eval_set = BRSpeechDataset(
        database_path,
        split=2,
        nb_samp=config["model_config"]["nb_samp"],
    )

    print("Training files:", len(train_set))
    print("Validation files:", len(dev_set))
    print("Test files:", len(eval_set))

    gen = torch.Generator()
    gen.manual_seed(seed)

    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
    )

    return trn_loader, dev_loader, eval_loader


# Whisper Encoding evaluation
def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2, thr[i]

def evaluate(loader, model, device):
    model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)

            _, logits = model(batch_x)

            probs = torch.softmax(logits, dim=1)[:, 1]

            scores.extend(probs.cpu().numpy())
            labels.extend(batch_y.numpy())

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    preds = (scores >= 0.5).astype(int)

    eer, eer_thr = compute_eer(labels, scores)

    print(f"Balanced accuracy : {balanced_accuracy_score(labels, preds):.4f}")
    print(f"ROC-AUC           : {roc_auc_score(labels, scores):.4f}")
    print(f"EER               : {eer:.4f} (threshold={eer_thr:.3f})")

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))

    print("\nClassification report:")
    print(classification_report(
        labels,
        preds,
        target_names=["real", "fake"]
    ))

    return {
        "scores": scores,
        "labels": labels,
        "predictions": preds,
        "eer": eer,
        "auc": roc_auc_score(labels, scores),
        "balanced_acc": balanced_accuracy_score(labels, preds),
    }


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    pbar = tqdm(
        trn_loader,
        desc="Training",
        total=len(trn_loader)
    )
    for batch_x, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))
        pbar.set_postfix(loss=batch_loss.item())

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
