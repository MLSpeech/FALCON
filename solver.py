import glob
from collections import OrderedDict, defaultdict
import os
import dill
import wandb
import torch_optimizer as optim_extra
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm

from dataloader import (TrainTestDataset,
                        TrainValTestDataset, collate_fn_padd, spectral_size)
from next_frame_classifier import NextFrameClassifier
from utils import (PrecisionRecallMetric, StatsMeter,
                   detect_peaks, line, max_min_norm, replicate_first_k_frames)

import dutch_preprocess
from utils import timit_to_leehon_map_MACRO


class Solver(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hp = cfg
        self.hp = hp
        self.hparams = hp
        self.current_epoch = 0
        
        self.peak_detection_params = defaultdict(lambda: {
            "width":      None,
            "distance":   None
        })
        self.pr = defaultdict(lambda: {
            "train": PrecisionRecallMetric(),
            "val":   PrecisionRecallMetric(),
            "test":  PrecisionRecallMetric()
        })
        
        self.best_l1_dist= defaultdict(lambda: {
            "train": (float("inf"), 0),
            "val":   (float("inf"), 0),
            "test":  (float("inf"), 0)
        })

        self.overall_best_l1_dist = 0
        self.stats = defaultdict(lambda: {
            "train": StatsMeter(),
            "val":   StatsMeter(),
            "test":  StatsMeter()
        })
        
        self.build_model()
        self.prepare_data()
        self.configure_optimizers()
    
    def build_model(self):
        print("MODEL:")
        self.NFC = NextFrameClassifier(self.hp)
        line()

    def prepare_data(self):
        if "timit" in self.hp.data and "buckeye" in self.hp.data:
            # joint training on TIMIT + Buckeye (data: timit_buckeye)
            t_train, t_val, t_test = TrainTestDataset.get_datasets(path=self.hp.timit_path)
            b_train, b_val, b_test = TrainValTestDataset.get_datasets(path=self.hp.buckeye_path, percent=self.hp.buckeye_percent)
            train = ConcatDataset([t_train, b_train])
            val   = ConcatDataset([t_val,   b_val])
            test  = ConcatDataset([t_test,  b_test])
            train.path = f"TIMIT+Buckeye train ({len(t_train)}+{len(b_train)})"
            val.path   = f"TIMIT+Buckeye val ({len(t_val)}+{len(b_val)})"
            test.path  = f"TIMIT+Buckeye test ({len(t_test)}+{len(b_test)})"
        elif "timit" in self.hp.data:
            train, val, test = TrainTestDataset.get_datasets(path=self.hp.timit_path)
        elif "buckeye" in self.hp.data:
            train, val, test = TrainValTestDataset.get_datasets(path=self.hp.buckeye_path, percent=self.hp.buckeye_percent)
        else:
            raise Exception("no such training data!")

        self.train_dataset = train
        self.valid_dataset = val
        self.test_dataset = test
        
        line()
        print("DATA:")
        print(f"train: {self.train_dataset.path} ({len(self.train_dataset)})")
        print(f"valid: {self.valid_dataset.path} ({len(self.valid_dataset)})")
        print(f"test: {self.test_dataset.path} ({len(self.test_dataset)})")
        line()
            
        _persistent = self.hp.dataloader_n_workers > 0
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.hp.batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn_padd,
                                       num_workers=self.hp.dataloader_n_workers,
                                       pin_memory=True,
                                       persistent_workers=_persistent)

        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.hp.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn_padd,
                                       num_workers=self.hp.dataloader_n_workers,
                                       pin_memory=True,
                                       persistent_workers=_persistent)

        self.test_loader  = DataLoader(self.test_dataset,
                                       batch_size=self.hp.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn_padd,
                                       num_workers=self.hp.dataloader_n_workers,
                                       pin_memory=True,
                                       persistent_workers=_persistent)

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.NFC.parameters())
        if self.hp.optimizer == "sgd":
            self.optimizer = optim.SGD(parameters, lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        elif self.hp.optimizer == "adam":
            self.optimizer = optim.Adam(parameters, lr=self.hparams.lr, weight_decay=5e-4)
        elif self.hp.optimizer == "adamW":
            self.optimizer = optim.AdamW(parameters, lr=self.hparams.lr)
        elif self.hp.optimizer == "ranger":
            self.optimizer = optim_extra.Ranger(parameters, lr=self.hparams.lr, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0)
        else:
            raise Exception("unknown optimizer")
        print(f"optimizer: {self.optimizer}")
        line()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=self.hp.lr_anneal_step,
                                                   gamma=self.hp.lr_anneal_gamma)
        return [self.optimizer]

    def train_one_epoch(self):
        self.run_epoch("train", self.current_epoch)
        return self.stats['nfc_loss']['train'].get_stats()
    
    def validate(self):
        self.evaluate("val", self.current_epoch)
        return self.stats['nfc_loss']['val'].get_stats()
    
    def run_epoch(self, mode, epoch):
        self.current_epoch = epoch
        is_train = mode=="train"
        loader = getattr(self, f"{mode}_loader")
        self.NFC.train(mode=="train")
        
        for batch_i, batch in enumerate(tqdm(loader)):
            if is_train:
                self.optimizer.zero_grad()
            result = self.forward(batch,batch_i,mode)
            
            if is_train:
                result['loss'].backward()
                self.optimizer.step()
        if is_train:
            self.scheduler.step()
        self.generic_eval_end(mode, epoch)

    
    def evaluate(self, mode, epoch=None):
        loader = self.valid_loader if mode == 'val' else self.test_loader
        self.NFC.eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(tqdm(loader)):
                self.forward(batch, batch_i, mode)
        self.generic_eval_end(mode, epoch if epoch is not None else self.current_epoch)
        
    def test(self):
        self.evaluate('test', epoch=-1)
    
    def forward(self, data_batch, batch_i, mode):
        loss = 0
        
        # TRAIN
        audio, seg, phonemes, length, fname = data_batch
        
        language = "english" # "dutch"
        if language == "dutch":
            lh39_ph = []
            for phoneme_seq in phonemes:
                lh39_ph_seq = []
                for IFA_ph in phoneme_seq:
                    print(f"\nINPUT: {IFA_ph}")
                    output = dutch_preprocess.aligner_pipeline(timit_to_leehon_map_MACRO[IFA_ph.lower()])
                    lh39_ph_seq.append([x["lh39"] for x in output])
                if not output:
                    print("Results: None")
                print(phoneme_seq)
                print(f"Dutch IPA to LH39 mapping: {lh39_ph_seq}")
                lh39_ph.append(np.hstack(lh39_ph_seq).tolist())
            print(f"Dutch IPA to LH39 total: {lh39_ph}")
            phonemes = lh39_ph
        
        
        
        
        if mode in ["test", "val"]: 
            with torch.no_grad():
                self.NFC.eval()
                preds,original_lengths, probs, frame_labels, _,preds_peaks, w_phi = self.NFC(audio,None,phonemes,length)
        else:
            self.NFC.train()
            preds,original_lengths, probs, frame_labels,_,preds_peaks, w_phi = self.NFC(audio,seg,phonemes,length)
        
        epoch = self.current_epoch
        if epoch < 5:
            total_loss, ph_loss, loss_nce, sum_mse, w_pos_neg, w_phi = self.NFC.loss_ph(preds,original_lengths, probs, frame_labels, seg,preds_peaks, w_phi, phonemes)
        else:
            total_loss, ph_loss, loss_nce, sum_mse, w_pos_neg, w_phi = self.NFC.total_loss(preds,original_lengths, probs, frame_labels, seg,preds_peaks, w_phi, phonemes)

        # InfoNCE LOSS ABLATION - 
        # total_loss, ph_loss, loss_nce, sum_mse, w_pos_neg, w_phi = self.NFC.loss_InfoNCE_classic(preds, original_lengths, probs, frame_labels, seg, preds_peaks, w_phi, phonemes)
        
        NFC_loss = total_loss
        
        self.stats['w_pos'][mode].update(torch.tensor(w_pos_neg[0].item()))
        self.stats['w_neg'][mode].update(torch.tensor(w_pos_neg[1].item()))
        self.stats['w_phi1'][mode].update(torch.tensor(w_phi[0].item()))
        self.stats['w_phi2'][mode].update(torch.tensor(w_phi[1].item()))
        self.stats['ph_loss'][mode].update(torch.tensor(ph_loss.item()))
        self.stats['nce_loss'][mode].update(torch.tensor(loss_nce.item()))
        self.stats['softDPmse_loss'][mode].update(torch.tensor(sum_mse.item()))
        self.stats['nfc_loss'][mode].update(torch.tensor(NFC_loss.item()))
        loss += NFC_loss        
        loss_key = "loss" if mode == "train" else f"{mode}_loss"
        return OrderedDict({
            loss_key: loss
        })

    def generic_eval_end(self, mode, epoch):
        metrics = {}
        data = self.hp.data

        for k, v in self.stats.items():
            metrics[f"{mode}_{k}"] = self.stats[k][mode].get_stats()
        metrics['epoch'] = epoch+1
        metrics['current_lr'] = self.optimizer.param_groups[0]['lr']

        line()
        # get best_l1_dist from all l1_dist types and all epochs
        best_overall_l1_dist = float("inf")
        for pred_type, l1_dist in self.best_l1_dist.items():
            if l1_dist[mode][0] < best_overall_l1_dist:
                best_overall_l1_dist = l1_dist[mode][0]
        metrics[f'{mode}_min_l1_dist'] = best_overall_l1_dist
        for k, v in metrics.items():
            print(f"\t{k:<30} -- {v}")
        line()
        wandb.log(metrics, step=epoch)

        output = OrderedDict({
            'log': metrics
        })

        return output
    
    def get_ckpt_path(self):
        # return glob.glob(self.hp.wd + "/*.ckpt")[0]
        return glob.glob(os.path.join(self.hp.wd + "/*.ckpt"))[0]