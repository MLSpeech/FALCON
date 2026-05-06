import os
import random
import socket
from argparse import Namespace
import wandb
import hydra
import numpy as np
import torch
from torch.backends import cudnn
import dill
from solver import Solver
import torch.cuda
from memory_profiler import profile

def set_cuda_visible_devices(device_ids):
    """
    Sets the CUDA_VISIBLE_DEVICES environment variable to the given device IDs.
-
    Args:
        device_ids (List[int]): List of GPU device IDs to make visible.
    """
    if not device_ids:
        raise ValueError("The device_ids list cannot be empty.")
    visible_devices = ",".join(map(str, device_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    print(f"CUDA_VISIBLE_DEVICES set to: {visible_devices}")

# torch.autograd.set_detect_anomaly(True)  # debug-only; ~2x slower. Re-enable to track NaNs.


@hydra.main(config_path='conf/config.yaml', strict=False)

def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    print(f"running in: {os.getcwd()}")
    cfg.wd = os.getcwd()
    cfg.host = socket.gethostname()
    cfg.project = "default" if not hasattr(cfg, "project") else cfg.project
    cfg = Namespace(**dict(cfg))

    # Initialize Weights & Biases here (rather than at module import) so
    # importing main.py doesn't trigger a W&B login. Project name is
    # configurable via the WANDB_PROJECT env var; disable entirely with
    # WANDB_MODE=disabled.
    wandb.init(project=os.environ.get("WANDB_PROJECT", "FDNFA"))
    
    set_cuda_visible_devices(cfg.devices)
    device = torch.device("cuda")
    
    solver = Solver(cfg).to(device)

    finetune_from = getattr(cfg, "finetune_from", None)
    start_epoch_override = getattr(cfg, "start_epoch", None)

    if finetune_from:
        # Transfer-learning seed: load model weights only, fresh optimizer,
        # skip warm-up curriculum by default.
        print(f"finetuning from {finetune_from}")
        ckpt = torch.load(finetune_from, map_location=device)
        solver.load_state_dict(ckpt['model_state_dict'])
        start_epoch = 5 if start_epoch_override is None else start_epoch_override
        ckpt_path = os.path.join(cfg.wd, "best_model.pt")  # placeholder; overwritten in loop
    elif cfg.ckpt is not None:
        ckpt_path = cfg.ckpt
    else:
        ckpt_path = os.path.join(cfg.wd, "best_model.pt")

    if not finetune_from:
        if os.path.exists(ckpt_path):
            print(f"loading ckpt from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            solver.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
        else:
            start_epoch = 0
        if start_epoch_override is not None:
            start_epoch = start_epoch_override

    if not cfg.ckpt:
        best_val_metric = float('-inf')
        for epoch in range(start_epoch, cfg.epochs):
            print(f"Epoch {epoch +1}/{cfg.epochs}")
            ckpt_path = os.path.join(cfg.wd, f"{epoch}_best_model.pt")
            solver.current_epoch = epoch
            train_loss = solver.train_one_epoch()
            val_metric = solver.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Val Metric: {val_metric:.4f}")
            print("New best model. Saving ckpt")
            print(f"new ckpt: {ckpt_path}")
            best_val_metric = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict' : solver.state_dict(),
                'optimizer_state_dict' : solver.optimizer.state_dict(),
                'hparams' : solver.hp,
                'peak_detection_params': dill.dumps(solver.peak_detection_params)
            }, ckpt_path)   
        
    print(f"Running test on ckpt: {ckpt_path}")
    print(f"Testing for {cfg.data.upper()}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        solver.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        solver.load_state_dict(ckpt['state_dict'])
    else:
        solver.load_state_dict(ckpt)
    solver.hp.timit_path = cfg.timit_path
    solver.hp.buckeye_path = cfg.buckeye_path
    solver.hp.data = cfg.data

    solver.test()

if __name__ == "__main__":
    main()