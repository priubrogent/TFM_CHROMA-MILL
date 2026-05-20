import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from models.archs import define_network
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('models.losses')
metric_module = importlib.import_module('metrics')

from metrics.psnr_ssim import calculate_psnr_batch_vectorized

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from data.transforms import wb
from torchvision.transforms.functional import adjust_gamma
from matplotlib import pyplot as plt
from torchvision import utils as vutils

import wandb


def _diff_to_viridis(gt_chw, pred_chw):
    """Absolute per-channel mean error mapped to viridis colormap. Returns 3×H×W float32 tensor."""
    diff = torch.abs(gt_chw - pred_chw).mean(0).numpy()
    rgb  = plt.cm.viridis(diff)[:, :, :3]
    return torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            print(f'Loading pretrained model [{load_path}] ...')
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        else: 
            print("No pretrained model found")

        if self.is_train:
            self.init_training_settings()

        try:
            self.l_sd = self.opt["train"]["losses"]["l_sd"]
            self.l_m = self.opt["train"]["losses"]["l_m"]
            self.l_chroma = self.out["train"]["losses"]["l_chroma"]
        except:
            self.l_sd = 0
            self.l_m = 0

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,self.opt['path'].get('strict_load_g',True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

        try:
            self.I = data['I'].to(self.device)
        except:
            self.I = None

        self.lq_path = data['lq_path']

        try:
            self.lq_gamma = data['lq_img_gamma']
            self.lq_original = data['lq_original']
        except:
            pass


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.lq_path = data['lq_path']

        try:
            self.I = data['I'].to(self.device)
        except:
            self.I = None

        try:
            self.lq_gamma = data['lq_img_gamma']
            self.lq_original = data['lq_original']
        except:
            pass


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # print(self.I, self.lq_path)
        # print("I: ", self.I, self.opt["datasets"]["train"]["feed_I"], self.opt["network_g"].get("use_I", False))
        if self.opt["datasets"]["train"]["feed_I"] or self.opt["network_g"].get("use_I", False):
            preds, illu_pred, illu_map, input_img = self.net_g(self.lq, self.I)
        else: 
            preds, illu_pred, chroma_pred, illu_map, input_img = self.net_g(self.lq)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]
        self.illu_pred = illu_pred

        loss_dict, loss_all = self.compute_loss(self.gt, preds, return_loss_all=True)
        loss_all.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        return loss_dict

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        
        self.net_g.eval()
        with torch.no_grad():
            if self.opt["datasets"]["train"]["feed_I"] or self.opt["network_g"].get("use_I", False):
                pred, illu_pred, illu_map, input_img = self.net_g(img, self.I)
                self.illu_pred = illu_pred
            else:
                pred, illu_pred, illu_map, input_img = self.net_g(img)
                self.illu_pred = illu_pred

        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred
        self.illu_map = illu_map
        self.mid_input_img = input_img
        self.net_g.train()

    def build_train_visual_grid(self, n_rows=5):
        """Build a wandb-ready grid from the last training batch: input | pred | GT | diff."""
        if not (hasattr(self, 'output') and hasattr(self, 'lq') and hasattr(self, 'gt')):
            return None
        lq   = self.lq.detach().cpu().clamp(0, 1)
        gt   = self.gt.detach().cpu().clamp(0, 1)
        pred = self.output.detach().cpu().clamp(0, 1)
        n_show = min(n_rows, lq.shape[0])
        tiles = []
        for i in range(n_show):
            diff = _diff_to_viridis(gt[i], pred[i])
            tiles += [lq[i], pred[i], gt[i], diff]
        return vutils.make_grid(tiles, nrow=4, padding=4, normalize=False)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def save_images_merged(self, current_iter, visuals, img_name, dataset_name):
        in_img = visuals['lq'].permute(0, 2, 3, 1)
        illu_map = self.illu_map.detach().cpu().clip(0, 1).permute(0, 2, 3, 1)
        mid_input_img = self.mid_input_img.detach().cpu().clip(0, 1).permute(0, 2, 3, 1)
        sr_img = visuals['result'].permute(0, 2, 3, 1)
        gt_img = visuals['gt'].permute(0, 2, 3, 1)
        predI = visuals["predI"].squeeze().detach().cpu().numpy()

        for i in range(in_img.shape[0]):
            name = osp.splitext(osp.basename(img_name[i]))[0]
            fig, ax = plt.subplots(1, 5, figsize=(64, 8))
            ax[0].imshow(in_img[i])
            try:
                ax[1].imshow(illu_map[i])
                ax[2].imshow(mid_input_img[i])
            except:
                pass
            ax[3].imshow(sr_img[i])
            ax[4].imshow(gt_img[i])
            ax[0].set_title(f"Input {int(name.split('_')[-1].split('.')[0])/10}", fontsize=24)
            ax[1].set_title("Illumination Map", fontsize=24)
            ax[2].set_title("Mid Input", fontsize=24)
            try:
                ax[3].set_title(f"Output {float(predI[i].mean()):.2f}", fontsize=24)
            except: 
                ax[3].set_title(f"Output {float(predI[i]):.2f}", fontsize=24)
            ax[4].set_title("GT", fontsize=24)
            ax[0].axis('off')   
            ax[1].axis('off')
            ax[2].axis('off')
            ax[3].axis('off')
            ax[4].axis('off')
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            os.makedirs(f"{self.opt['path']['visualization']}/{current_iter}/{dataset_name}", exist_ok=True)
            plt.savefig(f"{self.opt['path']['visualization']}/{current_iter}/{dataset_name}/{name}.png")
            plt.close()

    def save_images(self, img_name, current_iter, predI, in_img, sr_img, gt_img, lq_gamma=None):
        if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{current_iter}_{predI}.png')
            save_in_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_in.png')
            save_gt_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt.png')
        else:
            save_img_path = osp.join(self.opt['path']['test_images_dir'], f'{img_name}.png')
            save_gt_img_path = osp.join(self.opt['path']['test_images_dir'], f'{img_name}_gt.png')
            save_in_img_path = osp.join(self.opt['path']['test_images_dir'], f'{img_name}_in.png')

            try:
                save_gamma_img_path = osp.join(self.opt['path']['test_images_dir'], f'{img_name}_gamma.png')
                imwrite(lq_gamma, save_gamma_img_path)
            except:
                pass
            
            
        imwrite(sr_img, save_img_path)
        if current_iter is not None and current_iter < 10:
            imwrite(in_img, save_in_img_path)
            if gt_img is not None:
                imwrite(gt_img, save_gt_img_path)
    
    def change_gamma(self, train_gamma, out_gamma):
        val = out_gamma / train_gamma
        self.lq = adjust_gamma(self.lq, val)
        self.output = adjust_gamma(self.output, val)

        if hasattr(self, 'gt'):
            self.gt = adjust_gamma(self.gt, val)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, log=False):
        dataset_name = dataloader.dataset.opt['name'].split("Set")[0].lower()

        with_metrics = self.opt[dataset_name].get('metrics') is not None
        if with_metrics:
            self.metric_results = { metric: 0 for metric in self.opt[dataset_name]['metrics'].keys()}
        self.loss_val = OrderedDict()

        train_gamma = self.opt["datasets"][dataset_name]["gamma_train"]
        out_gamma = self.opt["datasets"][dataset_name]["gamma_out"]

        cnt = 0
        vis_tiles = []  # accumulate tiles for the validation visual grid

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.pad_test(self.opt[dataset_name]['window_size'])

            if train_gamma != out_gamma:
                print("Changing gamma from {} to {}".format(train_gamma, out_gamma))
                self.change_gamma(train_gamma, out_gamma)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            in_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr)

            gt_img = None
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)

            predI_tensor = visuals["predI"]
            if predI_tensor.dim() > 1:
                predI = str(predI_tensor.squeeze(1).detach().cpu().numpy())
            else:
                predI = str(predI_tensor.detach().cpu().numpy())

            try:
                lq_gamma = tensor2img([visuals['lq_gamma']], rgb2bgr=rgb2bgr)
            except:
                lq_gamma = None
                pass

            if save_img:
                self.save_images(img_name, current_iter, predI, in_img, sr_img, gt_img, lq_gamma=lq_gamma)

            # Collect one row per image (up to 5) for the validation grid
            if log and 'gt' in visuals and len(vis_tiles) // 4 < 5:
                lq_t   = visuals['lq'][0].clamp(0, 1)
                pred_t = visuals['result'][0].clamp(0, 1)
                gt_t   = visuals['gt'][0].clamp(0, 1)
                diff_t = _diff_to_viridis(gt_t, pred_t)
                vis_tiles += [lq_t, pred_t, gt_t, diff_t]

            if with_metrics:
                opt_metric = deepcopy(self.opt[dataset_name]['metrics'])
                if use_image: # if we want to use the tensor
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                
                else: # if we want to use the image
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'][0], visuals['gt'][0], **opt_)

            if dataset_name != "test":
                with torch.no_grad():
                    loss = self.compute_loss(self.gt, self.output, return_loss_all=False)
                for key, val in loss.items():
                    if key not in self.loss_val:
                        self.loss_val[key] = 0.
                    else:
                        self.loss_val[key] += val.detach().cpu().item()

            cnt += 1
            if hasattr(self, 'gt'):
                del self.gt
            del self.lq
            del self.output

        torch.cuda.empty_cache()

        # Log validation visual grid (input | pred | GT | diff, up to 5 rows)
        if log and vis_tiles:
            grid = vutils.make_grid(vis_tiles, nrow=4, padding=4, normalize=False)
            wandb.log({"val/visuals": wandb.Image(grid, caption='input | pred | GT | diff')}, step=current_iter)

        if dataset_name != "test":
            current_metric = 0.
            if with_metrics:
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= cnt
                    current_metric = self.metric_results[metric]

                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

            for key, val in self.loss_val.items():
                self.loss_val[key] /= cnt

        if hasattr(self, 'metric_results'):
            return self.metric_results, self.loss_val
        else:
            return None, self.loss_val


    
    def validate_small(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, log=False):
        dataset_name = dataloader.dataset.opt['name']
        opt = self.opt[dataset_name.split('-')[0]]
        
        self.metric_results = { metric: [] for metric in opt['metrics'].keys()}
        self.loss_val = OrderedDict()

        for idx, val_data in tqdm(enumerate(dataloader)):
            self.feed_data(val_data)
            self.pad_test(opt['window_size'])

            visuals = self.get_current_visuals()

            if save_img and idx <= 1:
                self.save_images_merged(current_iter, visuals, val_data['lq_path'], dataset_name)
            
            opt_metric = deepcopy(opt['metrics'])
            for name, opt_ in opt_metric.items():
                psnr = calculate_psnr_batch_vectorized(visuals['result'], visuals['gt'], **opt_)
                self.metric_results[name].append(psnr)

            loss = self.compute_loss(self.gt, self.output, return_loss_all=False)
            for key, val in loss.items():
                if key not in self.loss_val:
                    self.loss_val[key] = [0.]
                else:
                    self.loss_val[key].append(val.detach().cpu().item())

            del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

        for metric in self.metric_results.keys():
            self.metric_results[metric] = sum(self.metric_results[metric]) / len(self.metric_results[metric])
        
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        for key, val in self.loss_val.items():
            self.loss_val[key] = sum(val) / len(val)

        if hasattr(self, 'metric_results'):
            return self.metric_results, self.loss_val
        else:
            return None, self.loss_val



    def compute_loss(self, gt, preds, return_loss_all=False, I=None, miner=None, chroma_gt=None):
        loss_dict = OrderedDict()
        
        if not isinstance(preds, list):
            preds = [preds]
        
        l_pix = 0.
        if self.opt["train"]["losses"].get("weight_l1", False) and I is not None:
            for pred, i in zip(preds, I):
                weight_l1 = (1 - i)
                l_pix += self.cri_pix(pred, gt) * (weight_l1) * self.opt["train"]["losses"].get("l_pix", 1)
        else:
            for pred in preds:
                l_pix += self.cri_pix(pred, gt)

        loss_dict['l_pix'] = l_pix  * self.opt["train"]["losses"].get("l_pix", 1)
            
        if self.opt["datasets"]["train"]["pred_I"]:
            if I is None:
                real = self.I
            else:
                real = I


            # illu_pred may be [B] (already a per-image scalar) or [B,C,H,W]
            ndim = self.illu_pred.dim()
            if ndim >= 2:
                mean = self.illu_pred.mean(dim=list(range(1, ndim)))  # [B]
            elif ndim == 1:
                mean = self.illu_pred  # already [B]
            else:
                mean = self.illu_pred.unsqueeze(0)  # scalar → [1]

            illu_pred_mean = mean  # [B]
            real_target = real.to(self.illu_pred.device).view(-1)  # [B]
            difference = (illu_pred_mean - real_target) ** 2  # [B]
            abs_mean = torch.abs(difference)
            abs_difference_mean = abs_mean.mean()
            l_mse_illum = abs_difference_mean * self.l_m

            # l_mse_illum = torch.abs((self.illu_pred.mean(dim=(2,3)) - torch.ones_like(self.illu_pred.mean(dim=(2,3)))*real.unsqueeze(1).to(self.illu_pred.device)).float().mean()) * self.l_m
            # l_sd_illum = self.illu_pred.std(dim=(2, 3)).mean() 
            loss_dict['l_illum_pred'] = l_mse_illum
            # loss_dict['l_illum_sd'] = l_sd_illum

            loss_all = l_pix + l_mse_illum
        else:
            loss_all = l_pix


        # print("Loss: ", loss_dict)
        # exit(0)
        if self.opt["datasets"]["train"]["pred_chroma"] and hasattr(self, 'chroma_pred'):
            chroma_pred = self.chroma_pred # [B,2,32,32]
            pred = chroma_pred.mean(dim=(2,3)) # [B,2]
            real = chroma_gt.to(chroma_pred.device) # [B,2]

            l_chroma = F.mse_loss(pred, real) * self.l_chroma
            loss_dict['l_chroma'] = l_chroma
            loss_all += l_chroma
            # print("CHROMA PRED SHAPE: ", chroma_pred.shape)


        if return_loss_all:
            return loss_dict, loss_all
        else:
            return loss_dict

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {float(value):.4f}'
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu().clip(0, 1)
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        out_dict['predI'] = self.illu_pred.detach().cpu() # .mean(dim=(2,3)).detach().cpu()

        try:
            out_dict["lq_gamma"] = self.lq_gamma.detach().cpu()
            out_dict["lq_original"] = self.lq_original.detach().cpu()
        except:
            pass

        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        # if self.ema_decay > 0:
        #     path, name = self.save_network([self.net_g, self.net_g_ema],
        #                       'net_g',
        #                       self.optimizer_g,                                                                                                                            ,
        #                       param_key=['params', 'params_ema'])
        # else:
        path, name = self.save_network(self.net_g, self.optimizer_g, 'net_g', current_iter)
        # self.save_training_state(epoch, current_iter, **kwargs)
        return path, name 

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)

    def getEmbedding(self, dataloader):
        self.net_g.eval()

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            with torch.no_grad():
                pred, illu_pred, illu_map, input_img = self.net_g(self.lq, self.I)

            del self.lq
            torch.cuda.empty_cache()

        return illu_pred
