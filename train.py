
import argparse
import datetime
import logging
import math
import os

import random
import time
import torch
from os import path as osp

from data import create_dataloader, create_dataset
from data.data_sampler import EnlargedSampler
from data.data_util import collate_fn
from data.transforms import RandomBatchCrop
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import create_model
from utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from utils.dist_util import get_dist_info, init_dist
from utils.misc import mkdir_and_rename2
from utils.options import dict2str, parse
import torch
from pytorch_metric_learning import losses, miners, distances, testers

import numpy as np

from pdb import set_trace as stx
import wandb
from utils.our_utils import *

from base_parser import BaseParser
# from test import generate_test_images
from compute_metrics_chgs import *

def parse_options(is_train=True):
    wandb.login()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_name", default="run", help="name of test file") 
    # parser.add_argument("--params", default="1", help="params to load")
    parser.add_argument("--folder", default="zphone_linear_600x400_dif_m", help="folder to load setup from")
    args = parser.parse_args()

    # if args.params is None or args.params == "1": 
    opt = LoadParams(args.test_name, extra="/" + args.folder + "/")
    # elif args.params == "2":
    #     opt = LoadParams_no_prior(args.test_name)
    # elif args.params == "3":
    #     opt = LoadParams_feed_I(args.test_name)
    # elif args.params == "4":
    #     opt = LoadParams_feed_I_with_prior(args.test_name)

    gpu_list = ','.join(str(x) for x in opt["gpu_id"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    opt['dist'] = False
    print('Disable distributed.', flush=True)

    opt['rank'], opt['world_size'] = get_dist_info()

    seed = opt['manual_seed']
    set_random_seed(seed + opt['rank'])

    opt["path"]["visualization"] = opt["output_dir"]
    opt["path"]["experiments_root"] = opt["root_dir"]
    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['test_name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    log_file = osp.join(opt['path']['log'], f"metric.csv")
    logger_metric = get_root_logger(logger_name='metric', log_level=logging.INFO, log_file=log_file)
    metric_str = f'iter ({get_time_str()})'
    
    for k, v in opt['val']['metrics'].items():
        metric_str += f',{k}'
    
    logger_metric.info(metric_str)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = None
    if opt['logger']['use_tb_logger'] and 'debug' not in opt['test_name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt["path"]['experiments_root'], 'tb_logger'))
    return logger, tb_logger

def validate(model, val_loader, current_iter, tb_logger, opt, best_metric, epoch, log = True):
    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    save_img = epoch % opt['val']['save_img_freq'] == 0
    log_img = epoch % opt['val']['log_img_freq'] == 0
    current_metric, val_loss = model.validation(val_loader, current_iter, tb_logger, save_img, rgb2bgr, use_image = use_image, log = log_img)
    
    loss2log = {**dict(val_loss), **{"Total_loss": sum([l for n,l in val_loss.items() if n != "l_illum_sd"])}}

    logger_metric = get_root_logger(logger_name='metric')
    metric_str = f'{current_iter},{current_metric}'
    logger_metric.info(metric_str)

    if best_metric['psnr'] < current_metric['psnr']:
        best_metric['psnr'] = current_metric['psnr']
        best_metric['iter'] = current_iter
        model.save_best(best_metric)
    
    if tb_logger:
        tb_logger.add_scalar(f'metrics/best_iter', best_metric['iter'], current_iter)

        for k, v in opt['val']['metrics'].items(): 
            tb_logger.add_scalar(f'metrics/best_{k}', best_metric[k], current_iter)
    
    return current_metric, loss2log, best_metric

def compute_curriculum_prob(epoch, total_epochs, opt):
    cfg = opt.get("curriculum", {})
    if not cfg.get('enabled', False):
        return 1.0 # només LLIE
    start = cfg.get('start_epoch', 0)
    end = cfg.get('end_epoch', total_epochs)

    if epoch <= start: return 0.0
    if epoch >= end: return 1.0
    t = (epoch - start) / (end - start)
    if cfg.get('schedule', 'linear') == 'cosine':
        t = 0.5 * (1 - math.cos(math.pi * t))
    return t

def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase in ["train", "val"]:
            dataset_opt["phase"] = phase
            dataset_opt["scale"] = opt["scale"]
            if phase == 'train':
                dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)

                train_set = create_dataset(dataset_opt)
                train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
                train_loader = create_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'], collate_fn= lambda batch: collate_fn(batch, RandomBatchCrop(128), phase='train'))

                num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
                total_epochs = int(opt["datasets"]['train']["total_epochs"])
                total_iters = int(num_iter_per_epoch * total_epochs)
                logger.info(
                    'Training statistics:'
                    f'\n\tNumber of train images: {len(train_set)}'
                    f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                    f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                    f'\n\tWorld size (gpu number): {opt["world_size"]}'
                    f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                    f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

            elif phase == 'val':
                val_set = create_dataset(dataset_opt)
                val_loader = create_dataloader(val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
                logger.info(
                    f'Number of val images/folders in {dataset_opt["name"]}: '
                    f'{len(val_set)}')
            
        elif phase == 'test':
            continue
        
        else:
            print(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def log_artifact(path, model_name, metric_val):
    artifact = wandb.Artifact(model_name, type='model', metadata=metric_val)
    artifact.add_file(path)
    wandb.run.log_artifact(artifact)  
    wandb.save(path)


def train_batch(model, train_data, current_iter, opt, train_epoch_loss, epoch, iter_time, data_time, msg_logger, logger):    
    model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
    
    model.feed_train_data(train_data)
    loss = model.optimize_parameters(current_iter)

    loss2log = {**dict(loss), **{"Total_loss": sum([l for n,l in loss.items() if n != "l_illum_sd"])}}
    # print(model.get_current_learning_rate())
    try:
        wandb.log({"train_batch_loss":loss2log, "epoch": epoch, "lr":float(model.get_current_learning_rate()[0]), "mined_triplets": model.miner.num_triplets}, step=current_iter)
    except:
        wandb.log({"train_batch_loss":loss2log, "epoch": epoch, "lr":float(model.get_current_learning_rate()[0])}, step=current_iter)

    for k, v in loss2log.items():
        if k not in train_epoch_loss:
            train_epoch_loss[k] = 0
        train_epoch_loss[k] += v

    iter_time = time.time() - iter_time
    
    if current_iter % opt['logger']['print_freq'] == 0:
        log_vars = {'epoch': epoch, 'iter': current_iter}
        log_vars.update({'lrs': model.get_current_learning_rate()})
        log_vars.update({'time': iter_time, 'data_time': data_time})
        log_vars.update(model.get_current_log())
        msg_logger(log_vars)

    # save models and training states
    if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
        logger.info('Saving models and training states.')
        path, name = model.save(epoch, current_iter)
        # save the state of everything to wandb
        log_artifact(path, name, loss2log)


    data_time = time.time()
    iter_time = time.time()
    return data_time, iter_time

            
def main():
    opt = parse_options(is_train=True)
    opt["is_train"] = True

    torch.backends.cudnn.benchmark = True
    
    # make_exp_dirs(opt)
    if opt['logger']['use_tb_logger'] and 'debug' not in opt['test_name'] and opt['rank'] == 0:
        mkdir_and_rename2(osp.join(opt["path"]["experiments_root"], 'tb_logger'), opt['rename_flag'])
        opt["path"]["log"] = opt["root_dir"] + '/tb_logger/'

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # initialize triplet miners
    if opt["train"].get("losses", False) and opt["train"]["losses"].get("use_miner", False):
        miner_e = miners.TripletMarginMiner(margin=opt["train"]["losses"]["m"], type_of_triplets="easy")
        miner_s = miners.TripletMarginMiner(margin=opt["train"]["losses"]["m"], type_of_triplets="semihard")
        miner_h = miners.TripletMarginMiner(margin=opt["train"]["losses"]["m"], type_of_triplets="hard")
        miner_a = miners.TripletMarginMiner(margin=opt["train"]["losses"]["m"], type_of_triplets="all")

    wb_project = opt["wandb"].get("project", "Retinexformer")
    wb_entity = opt["wandb"].get("entity", None)

    if opt["wandb"]["resume"] == "must":
        wandb.init(project=wb_project, entity=wb_entity, name=opt["test_name"], config=opt, id=opt["wandb"]["id"], resume="must")
        path, current_iter = find_last_weights(opt)
        opt["path"]["pretrain_network_g"] = path
        start_epoch = current_iter // len(train_loader)

    else:
        wandb.init(project=wb_project, entity=wb_entity, name=opt["test_name"], config=opt)
        start_epoch = 0
        current_iter = 0

    model = create_model(opt)

    if opt["wandb"]["resume"] == "must":
        temp = 0
        for _ in range(start_epoch):
            model.update_learning_rate(temp, warmup_iter=opt['train'].get('warmup_iter', -1))
            temp += opt["datasets"]["train"]["batch_size_per_gpu"]

    best_metric = {'iter': 0, 'psnr': 0}
    for k, v in opt['val']['metrics'].items():
        best_metric[k] = 0

    if opt["wandb"]["resume"] == "must":
        best_psnr_path = osp.join(opt['path']['experiments_root'], 'best_psnr.pth')
        if osp.exists(best_psnr_path):
            best_ckpt = torch.load(best_psnr_path, map_location='cpu', weights_only=False)
            if 'best_metric' in best_ckpt:
                best_metric = best_ckpt['best_metric']

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    gt_size = opt['datasets']['train'].get('gt_size')

    scale = opt['scale']
    
    numEpochsNotImproved = 0
    prevValLoss = [10000000]
    epsilon = 0.0025
    total2stop = 100000
    try: 
        model.set_miner(miner_h)
    except: 
        pass
    # opt["train"]["losses"]["l_pix"] = 0
    for epoch in range(start_epoch, total_epochs + 1):
        # if epoch > 50: 
        #     opt["train"]["losses"]["l_pix"] = 5
        # elif epoch > 2000: 
            # model.set_miner(miner_h)

        train_epoch_loss = {}
        train_sampler.set_epoch(epoch)
        curriculum_prob = compute_curriculum_prob(epoch, total_epochs, opt)
        if hasattr(train_loader.dataset, 'set_curriculum_prob'):
            train_loader.dataset.set_curriculum_prob(curriculum_prob)
        wandb.log({"curriculum_prob": curriculum_prob}, step=current_iter)
        prefetcher.reset()
        train_data = prefetcher.next()
        data2log = {}
        total_mined_triplets = 0
        while train_data is not None:
            data_time = time.time() - data_time
            iter_time, data_time = train_batch(model, train_data, current_iter, opt, train_epoch_loss, epoch, iter_time, data_time, msg_logger, logger)
            train_data = prefetcher.next()
            current_iter += 1
            try: 
                total_mined_triplets += model.miner.num_triplets
            except: 
                pass

        # Log train visual grid (input | pred | GT | diff, up to 5 rows)
        if epoch % opt['val'].get('log_img_freq', 500) == 0:
            train_grid = model.build_train_visual_grid(n_rows=5)
            if train_grid is not None:
                wandb.log({"train/visuals": wandb.Image(train_grid, caption='input | pred | GT | diff')}, step=current_iter)

        if opt.get('val') is not None and (epoch % opt['val']['val_freq'] == 0):
            val_metrics, val_epoch_loss, best_metric = validate(model, val_loader, current_iter, tb_logger, opt, best_metric, epoch, log=True)
            wandb.log({"val_epoch_loss":val_epoch_loss, "val_metrics": val_metrics}, step=current_iter)

        # if min(prevValLoss) <= val_metrics["psnr"]:
        #     numEpochsNotImproved += 1
        # else:
        #     numEpochsNotImproved = 0

        # prevValLoss.append( val_metrics["psnr"])

        train_epoch_loss = {k: v / len(train_loader) for k, v in train_epoch_loss.items()}
        time_taken = time.time() - start_time
        # wandb.log({"epoch":epoch, "train_epoch_loss":train_epoch_loss, "time_taken":time_taken, "earlystopping": total2stop - numEpochsNotImproved, "mined_triplets": total_mined_triplets}, step=current_iter)
        wandb.log({"epoch":epoch, "train_epoch_loss":train_epoch_loss, "time_taken":time_taken, "mined_triplets": total_mined_triplets}, step=current_iter)
        
        # if numEpochsNotImproved >= total2stop:
        #     print("Early stopping at epoch: ", epoch)
        #     break

        
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    model.save(epoch=-1, current_iter=-1)
    
    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    evaluate_all(opt, model)

    wandb.finish()


if __name__ == '__main__':
    main()
