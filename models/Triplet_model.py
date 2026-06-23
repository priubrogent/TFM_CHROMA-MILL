import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from torchvision.transforms.functional import adjust_gamma

from models.archs import define_network
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('models.losses')
metric_module = importlib.import_module('metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

from models.image_restoration_model import *



class TripletModel(ImageCleanModel):
    def __init__(self, opt):
        super(TripletModel, self).__init__(opt)
        self.l_tr = self.opt["train"]["losses"]["l_tr"]
        self.miner = None

    def init_training_settings(self):
        self.opt['train']['ema_decay'] = 0
        super().init_training_settings()
        
        self.TripletLoss = torch.nn.TripletMarginLoss(margin=self.opt["train"]["losses"].get("m", 1.0), p=2)
        print("Triplet Loss Margin: ", self.opt["train"]["losses"].get("m", 1.0))

    def feed_train_data(self, data):
        self.anchor = data['anchor']
        self.pos = data['positive']
        self.neg = data['negative']

        # send all the values from each dictionary to th device if necessary
        for k, v in self.anchor.items():
            if isinstance(v, torch.Tensor):
                self.anchor[k] = v.to(self.device)
        
        for k, v in self.pos.items():
            if isinstance(v, torch.Tensor):
                self.pos[k] = v.to(self.device)

        for k, v in self.neg.items():
            if isinstance(v, torch.Tensor):
                self.neg[k] = v.to(self.device)


    def feed_data(self, data):
        super().feed_data(data)
        if self.opt["datasets"]["train"]["feed_I"]:
            self.anchor = {"lq": self.lq, "gt": self.gt, "lq_path": self.lq_path, "I": self.I}
        else:
            self.anchor = {"lq": self.lq, "gt": self.gt, "lq_path": self.lq_path, "I": None}

        if 'chroma_gt' in data:
            v = data['chroma_gt']
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            self.anchor["chroma_gt"] = v.to(self.device)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if self.opt["datasets"]["train"]["feed_I"]:
            preds, self.illu_pred, self.chroma_pred, embedding, self.illu_pred_mlp = self.net_g(self.anchor["lq"], self.anchor["I"])
            preds_pos, illu_pred_pos, chroma_pred_pos, embedding_pos, _ = self.net_g(self.pos["lq"], self.pos["I"])
            preds_neg, illu_pred_neg, chroma_pred_neg, embedding_neg, _ = self.net_g(self.neg["lq"], self.neg["I"])

        else:
            preds, self.illu_pred, self.chroma_pred, embedding, self.illu_pred_mlp = self.net_g(self.anchor["lq"])
            preds_pos, illu_pred_pos, chroma_pred_pos, embedding_pos, _ = self.net_g(self.pos["lq"])
            preds_neg, illu_pred_neg, chroma_pred_neg, embedding_neg, _ = self.net_g(self.neg["lq"])

        if not isinstance(preds, list):
            preds = [preds]

        self.output_decoder = preds[-1]  # decoder output before gain (for visualization)

        # Reconstruction loss: decoder output (before gain) should match the input image
        loss_dict = OrderedDict()
        loss_all = 0
        if self.l_recon > 0:
            l_recon = 0
            for pred in (preds if isinstance(preds, list) else [preds]):
                l_recon += self.cri_pix(pred, self.anchor["lq"])
            l_recon = l_recon * self.l_recon
            loss_dict['l_recon'] = l_recon
            loss_all += l_recon

        # Optional: correct image brightness using predicted intensity before loss
        if self.opt["train"]["losses"].get("correct_intensity", False) and self.illu_pred_mlp is not None:
            gain = 1.0 / (self.illu_pred_mlp.view(-1, 1, 1, 1) + 1e-6)
            preds = [torch.clamp(p * gain, 0.0, 1.0) for p in preds]

        self.output = preds[-1]  # corrected output (for parent class compatibility)
        self.output_corrected = preds[-1]  # corrected output (for visualization)

        # Compute remaining losses on corrected predictions
        loss_dict_main, loss_all_main = self.compute_loss(self.anchor["gt"], preds, embedding, embedding_pos, embedding_neg, return_loss_all=True)
        loss_dict.update(loss_dict_main)
        loss_all += loss_all_main
        loss_all.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)        
        return loss_dict

    def compute_loss(self, gt, preds, embedding=None, embedding_pos=None, embedding_neg=None, return_loss_all=False):
        if "I" in self.anchor.keys():
            I = self.anchor["I"]
        else:
            I = None
        
        if "chroma_gt" in self.anchor.keys():
            chroma_gt = self.anchor["chroma_gt"]
        else:
            chroma_gt = None
        
        loss_dict, loss_all = super().compute_loss(gt, preds, return_loss_all=True, I=I, chroma_gt=chroma_gt)

        # indices_tuple = self.miner(embedding, embedding_pos, embedding_neg)
        if embedding is not None:
            loss_dict["l_triplet"] = self.TripletLoss(embedding, embedding_pos, embedding_neg) * self.l_tr
            loss_all += loss_dict["l_triplet"]

        if return_loss_all:
            return loss_dict, loss_all
        else:
            return loss_dict

    def build_train_visual_grid(self, n_rows=5):
        """Build a wandb-ready grid: input | decoder output | corrected | GT."""
        if not (hasattr(self, 'output_decoder') and hasattr(self, 'output_corrected') and hasattr(self, 'anchor')):
            return None
        lq        = self.anchor["lq"].detach().cpu().clamp(0, 1)
        decoder   = self.output_decoder.detach().cpu().clamp(0, 1)
        corrected = self.output_corrected.detach().cpu().clamp(0, 1)
        gt        = self.anchor["gt"].detach().cpu().clamp(0, 1)
        n_show = min(n_rows, lq.shape[0])
        tiles = []
        for i in range(n_show):
            tiles += [lq[i], decoder[i], corrected[i], gt[i]]
        return vutils.make_grid(tiles, nrow=4, padding=4, normalize=False)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.anchor["lq"], (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.anchor["lq"]
        self.net_g.eval()
        with torch.no_grad():
            if self.opt["datasets"]["train"]["feed_I"]:
                pred, self.illu_pred, self.chroma_pred, self.embedding, _ = self.net_g(img, self.anchor["I"])
            else:
                pred, self.illu_pred, self.chroma_pred, self.embedding, _ = self.net_g(img)

        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred
        self.net_g.train()

    def get_embedding(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        window_size = self.opt['val'].get('window_size', 0) # 4 in val

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        all_embeddings = torch.tensor([]).to(self.device)
        all_names = []
        all_scenes = []
        for idx, val_data in tqdm(enumerate(dataloader)):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            scene = img_name.split("-")[0]

            self.feed_data(val_data)
            test()

            all_embeddings = torch.cat((all_embeddings, self.embedding), dim=0)
            all_names.append(img_name)
            all_scenes.append(scene)

        return all_embeddings, all_names, all_scenes
    
    def set_miner(self, miner):
        self.miner = miner