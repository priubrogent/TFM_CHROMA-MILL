import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchsummary import summary
import cv2
import torchvision.transforms.functional as TF
import PIL  
from PIL import Image
import os
from natsort import natsorted
from glob import glob
import argparse
import pyiqa
import wandb
from utils.our_utils import *
from base_parser import BaseParser
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import deltaE_cie76
from skimage.metrics import mean_squared_error as MSE
import logging
import torch
from os import path as osp

from data import create_dataloader, create_dataset
from models import create_model
from utils import (get_env_info, get_root_logger, get_time_str,make_exp_dirs)
from utils.options import dict2str
import wandb

from utils.our_utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
ssim = pyiqa.create_metric('ssim').to(device)
lpips = pyiqa.create_metric('lpips').to(device)
niqe = pyiqa.create_metric('niqe').to(device)
brisque = pyiqa.create_metric('brisque').to(device)
hyperiqa = pyiqa.create_metric('hyperiqa').to(device)


def generate_test_images(opt, model):
    test_loaders = []
    opt["phase"] = phase = 'test'
    dataset_opt = opt['datasets'][phase]

    dataset_opt["phase"] = phase
    dataset_opt["scale"] = opt["scale"]

    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
    print(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    test_loaders.append([test_loader, phase])

    # create model
    for test_loader, phase in test_loaders:
        print(f'Testing {phase}...')
        rgb2bgr = opt[phase].get('rgb2bgr', True)
        use_image = opt[phase].get('use_image', True)
        metrics = model.validation(test_loader, current_iter=0, tb_logger=None, save_img=opt['val']['save_img'], rgb2bgr=rgb2bgr, use_image=use_image)

def initialize(test_name, extra = ""):    
    opt = LoadParams(test_name, extra = extra)
    opt["path"]["visualization"] = opt["output_dir"]
    opt["path"]["experiments_root"] = opt["root_dir"]
    
    if opt['logger']['use_tb_logger'] and 'debug' not in opt['test_name'] and opt['rank'] == 0:
        opt["path"]["log"] = opt["root_dir"] + '/tb_logger/'

    # opt["path"]["pretrain_network_g"] = None

    opt["is_train"] = False
    opt["dist"] = False

    model = create_model(opt)
    print("model created")

    initialize_test_params(opt, model, dsName = "nikon", path = None, original = "original" in opt["datasets"]["root"])
    return opt, model

def initialize_test_params(opt, model, dsName = "nikon", path = None, original = False):
    opt["path"]["test_images_dir"] = opt["root_dir"] + "/Test_images_" +  "%.2f" % opt["datasets"]["test"]["gamma_train"] + "_" + "%.2f" % opt["datasets"]["test"]["gamma_out"] + "_" + dsName + "/"
    print(opt["path"]["test_images_dir"])
    
    opt["datasets"]["test"]["dataroot_gt"] = get_proper_path(opt, dsName)
    opt["datasets"]["test"]["dataroot_lq"] = opt["datasets"]["test"]["dataroot_gt"].replace("high", "low")

    if dsName == "lol":
        opt["datasets"]["test"]["gamma_in"] = 1/2.2
        opt["datasets"]["test"]["gamma_train"] = 1/2.2
        opt["datasets"]["test"]["gamma_out"] = 1/2.2

        opt["datasets"]["test"]["type"] = "Dataset_PairedImage"

    opt["is_train"] = False
    opt["dist"] = False

    if opt["datasets"]["test"]["io_backend"].get("type", None) == None:
        opt["datasets"]["test"]["io_backend"]["type"] = "disk"

    # try:
    load_path, iter = find_best_val_weights(opt)
    opt["path"]["pretrain_network_g"] = path 
    print(f'Loading pretrained model [{load_path}] ...')
    model.load_network(model.net_g, load_path, opt['path'].get('strict_load_g', True), param_key=opt['path'].get('param_key', 'params'))
    # except:
    #     print("Could not load the weights")
    #     pass

    return opt, model

def get_proper_path(opt, dsName):
    root = "/".join(opt["datasets"]["root"].split("/")[:-2])
    
    if dsName == "phone":
        return root + "/phone/test/high/"
        
    elif dsName == "nikon":
        return root + "/nikon/test/high/"
    
    elif dsName == "lol":
        return "/ghome/mpilligua/lowlight/LOLdataset/eval15/high/"




def compute_metrics2(path_pred, path_true, path_original, dsName, plot = True):
    metrics = {}
    y_pred = cv2.imread(path_pred)
    y_true = cv2.imread(path_true)
    y_original = cv2.imread(path_original)
    # print(y_pred.shape, y_true.shape, y_original.shape)

    global psnr, ssim, lpips, niqe, brisque, hyperiqa

    metrics['MSE ↓'] = np.mean((y_pred - y_true) ** 2)
    metrics['PSNR ↑'] = psnr(path_pred, path_true).item()
    metrics['SSIM ↑'] = ssim(path_pred, path_true).item()
    metrics['LPIPS ↑'] = lpips(path_pred, path_true).item()
    metrics['DeltaE ↓'] = np.mean(deltaE_cie76(color.rgb2lab(y_pred), color.rgb2lab(y_true)))
    metrics['NIQE ↓'] = niqe(path_pred).item()
    metrics['BRISQUE ↓'] = brisque(path_pred).item()
    metrics['HyperlQA ↑'] = hyperiqa(path_pred).item()


    if plot:
      fig, ax = plt.subplots(1, 3, figsize=(12, 4))
      y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)
      y_original = cv2.cvtColor(y_original, cv2.COLOR_BGR2RGB)
      y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2RGB)

      ax[2].imshow(y_true)
      ax[0].imshow(y_original)
      ax[1].imshow(y_pred)

      ax[2].set_title("GT")
      ax[0].set_title("Original")
      ax[1].set_title("Inferenced")

      ax[0].axis("off")
      ax[1].axis("off")
      ax[2].axis("off")

      plt.show()

    #   display(pd.DataFrame.from_dict({k:[round(v, 4)] for k,v in metrics.items()}))
    return metrics

def plot_metrics(list_of_results, test_name, opt, dsName, separated = False):
    out_dir = opt["root_dir"]
    createDir(out_dir + "/results")    
    createDir(out_dir + "/metrics_" + dsName + "/")   

    img2print = len(list_of_results)
    if img2print > 40:
        img2print = 40

    print("Ploting {} images".format(img2print))

    if len(list_of_results[0]) == 6:
        fig, ax = plt.subplots(img2print, 4, figsize=(12, 2*img2print))
        if opt["datasets"]["val"]["WB"] == "aft":
            ax[0, 0].set_title("Original")
            ax[0, 1].set_title("Inferenced")
            # ax[0, 2].set_title("Gamma")
            ax[0, 2].set_title("WB")
            ax[0, 3].set_title("GT")
        else: 
            ax[0, 0].set_title("Original")
            # ax[0, 1].set_title("Gamma")
            ax[0, 1].set_title("WB")
            ax[0, 2].set_title("Inferenced")
            ax[0, 3].set_title("GT")
    else:
        fig, ax = plt.subplots(img2print, 3, figsize=(8, 2*img2print))
        ax[0, 2].set_title("GT")
        ax[0, 0].set_title("Original")
        ax[0, 1].set_title("Inferenced")

    list_of_results = natsorted(list_of_results, key=lambda x: x[3].split("B_")[-1].split("_")[0])

    # print(len(list_of_results[0]))
    metrics = []
    for i, data in enumerate(list_of_results):
        try:
            if opt["datasets"]["val"]["WB"] == "aft":
                pred, high, low, name, gamma, wb = data
            else: 
                pred, high, low, name, gamma, original = data
        except:
            pred, high, low, name = data

        pred.save(out_dir + "/results/pred.png")
        high.save(out_dir + "/results/high.png")
        low.save(out_dir + "/results/low.png")

        pred = cv2.imread(out_dir + "/results/pred.png")
        high = cv2.imread(out_dir + "/results/high.png")
        low = cv2.imread(out_dir + "/results/low.png")
        if i < img2print:
            y_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            y_high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)
            y_low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)

            ax[i, 2].imshow(y_high)
            ax[i, 0].imshow(y_low)
            ax[i, 1].imshow(y_pred)

            ax[i, 0].get_xaxis().set_ticks([])
            ax[i, 0].get_yaxis().set_ticks([])
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")

        name = name.split("-")[0] + "-" + name.split("-")[-1]

        if i < img2print:
            ax[i, 0].set_ylabel(name)

        metrics.append(compute_metrics2(out_dir + "/results/pred.png", out_dir + "/results/high.png", out_dir + "/results/low.png", plot = False, dsName = dsName))

    # do the mean of the metrics of all the images
    # print(len(metrics))
    sorted_metrics = {k:[] for k in metrics[0].keys()}
    for item in metrics:
        for k, v in item.items():
            sorted_metrics[k].append(v)
    
    sorted_metrics = {**{"name": test_name}, **{k:np.mean(v) for k,v in sorted_metrics.items()}}


    # print(sorted_metrics)
    metrics_df = pd.DataFrame.from_dict({k:[v] for k,v in sorted_metrics.items()}).T
    # print(metrics_df)
    # print(metrics_df.index)
    # print(metrics_df.values)

    wandb.log({"summary": wandb.Table(dataframe=metrics_df.T, columns=list(metrics_df.columns))})

    if not separated:
        metrics_df.to_csv(out_dir + "/metrics_" + dsName + "/metrics.csv")
        try:
            plt.savefig(out_dir + "/metrics_" + dsName + "/inference.png", dpi=300, bbox_inches='tight')
            wandb.log({"inference": wandb.Image(out_dir + "/metrics_" + dsName + "/inference.png")})
        except:
            print("Could not save the inference image of all the images")
            pass
    
    else:
        metrics_df.to_csv(out_dir + "/metrics_" + dsName + "/{}_metrics.csv".format(test_name))
        try:
            plt.savefig(out_dir + "/metrics_" + dsName + "/{}_inference.png".format(test_name), dpi=300, bbox_inches='tight')
            wandb.log({"inference_{}".format(test_name): wandb.Image(out_dir + "/metrics_" + dsName + "/{}_inference.png".format(test_name))})
        except:
            print("Could not save the inference image of intensity {}".format(test_name))
            pass

def separate_by_intensity(results, opt, dsName):
    out_dir = opt["root_dir"]
    ListPerI = {}

    metrics = []
    for i, data in enumerate(results):
        try:
            pred, high, low, name, gamma, wb = data
        except:
            pred, high, low, name = data

        intensities = int(name.split("B_")[1].split(".")[0])
        if intensities not in ListPerI:
            ListPerI[intensities] = []
        
        ListPerI[intensities].append(data)

    for k, v in ListPerI.items():
        plot_metrics(v, "I_" + str(k).rjust(3, "0"), opt, separated = True, dsName = dsName)

def create_table(opt, dsName):
    out_dir = opt["root_dir"]
    all_df = pd.DataFrame()
    for file in os.listdir(out_dir + "/metrics_" + dsName + "/", dsName):
        if "metrics" in file:
            df = pd.read_csv(out_dir + "/metrics_" + dsName + "/" + file).T
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            pd.DataFrame.set_index(df, "name")
            all_df = all_df._append(df)

    all_df = all_df.sort_values(by=['name'])

    # pass to float and round to 2 decimals ignoring the name column
    all_df[all_df.columns[1:]] = all_df[all_df.columns[1:]].astype(float).round(2)
    print(all_df)

    table = all_df

    create_plots(all_df, opt, dsName)

    #save it using matplotlib
    fig, ax = plt.subplots(figsize=(1*len(table.columns), 3))
    ax.axis('off')

    ax.axis('tight')
    ax.table(cellText=table.values, colLabels=table.columns, loc='center', cellLoc='center', colLoc='center', rowLabels=table.index)
    fig.tight_layout()
    plt.savefig(out_dir + "/metrics_" + dsName + "/table.png", dpi=300, bbox_inches='tight')
    # wandb.log({"table": wandb.Image(out_dir + "/metrics_" + dsName + "/table.png")})

    # # create a table that only has PSNR, SSIM, NIQUE and DeltaE
    table = all_df[["PSNR ↑", "SSIM ↑", "NIQE ↓", "DeltaE ↓"]]
    table = table.sort_values(by=['name'])

    #save it using matplotlib
    fig, ax = plt.subplots(figsize=(1*len(table.columns), 3))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=table.values, colLabels=table.columns, loc='center', cellLoc='center', colLoc='center', rowLabels=table.index)
    fig.tight_layout()
    plt.savefig(out_dir + "/metrics_" + dsName + "/table2.png", dpi=300, bbox_inches='tight')
    # wandb.log({"table2": wandb.Image(out_dir + "/metrics_" + dsName + "/table2.png")})

def create_plots(df, opt, dsName):
    df.set_index("name", inplace=True)

    df.index = df.index.str.split("_").str[-1]
    df = df[df.index.str.isdigit()]

    df = df.astype(float)

    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(20, 10)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, metric in enumerate(df.columns):
        df[metric].plot(ax=ax[i//4, i%4], title=metric)
        ax[i//4, i%4].set_ylabel("Value")
        ax[i//4, i%4].set_xlabel("Intensity")

    plt.savefig(opt["root_dir"] + "/metrics_" + dsName + "/plot.png", dpi=300, bbox_inches='tight')
    # if args.wandb:
    #     wandb.log({"plot": wandb.Image(out_dir + "/metrics_" + dsName + "/plot.png")})

I2sensor = {'Red': {254: 1002.4110000000001, 241: 902.8583, 227: 801.4689, 213: 706.1163, 197: 604.5339, 180: 505.242, 161: 404.8023, 139: 302.3945, 113: 200.58630000000002, 80: 101.352, 0: 49}, 'Green': {254: 1926.4376, 241: 1734.1636999999998, 227: 1538.4017000000001, 213: 1354.3604999999998, 197: 1158.3797, 180: 966.924, 161: 773.3957, 139: 576.2801, 113: 380.6405, 80: 190.54399999999998, 0: 49}, 'Blue': {254: 193.59879999999998, 241: 173.97789999999998, 227: 154.0195, 213: 135.2763, 197: 115.34349999999999, 180: 95.904, 161: 76.2979, 139: 56.3923, 113: 36.7363, 80: 17.823999999999998, 0: 49}, 'White': {254: 1861.312, 241: 1675.5043, 227: 1486.3278999999998, 213: 1308.4803, 197: 1119.0979, 180: 934.0919999999999, 161: 747.0883, 139: 556.6255, 113: 367.60029999999995, 80: 183.95199999999997, 0: 49}}
IList = [0, 80, 113, 139, 161, 180, 197, 213, 227, 241, 254]

def create_wandbPlots(opt, dsName):
    out_dir = opt["root_dir"]
    df = pd.DataFrame()
    for file in os.listdir(out_dir + "/metrics_" + dsName + "/"):
        if file.endswith(".csv"):
            try:
                I = file.split("_")[-2]

                df_tmp = pd.read_csv(out_dir + "/metrics_" + dsName + "/" + file)
                df_tmp = df_tmp.set_index("Unnamed: 0")
                df_tmp = df_tmp.rename(columns={"Unnamed: 0": "I"})
                df_tmp = df_tmp.T
                df_tmp["I"] = (int(I)/254 * 100)
                df_tmp["I_lights"] = int(I)
                df_tmp["I_sensors"] = I2sensor["White"][int(I)]
                df_tmp["I_linear"] = IList.index(int(I))*10
                df_tmp.drop(columns=["name"], inplace=True)
                df_tmp = df_tmp.astype(float)
                df_tmp.set_index("I", inplace=True)

                df = df._append(df_tmp)
                df.sort_index(inplace=True)

            except IndexError:
                continue
    
    print(df)
    table = wandb.Table(dataframe = df.reset_index())
    wandb.log({f"{dsName.capitalize()}-Inference/Metrics": table})
    # wandb.log({"PSNR" : wandb.plot.line(table, "I", "PSNR ↑", title="PSNR per Intensity")})
    # wandb.log({"DeltaE" : wandb.plot.line(table, "I", "DeltaE ↓", title="DeltaE per Intensity")})
    # wandb.log({"LPIPS" : wandb.plot.line(table, "I", "LPIPS ↑", title="LPIPS per Intensity")})
    # wandb.log({"MSE" : wandb.plot.line(table, "I", "MSE ↓", title="MSE per Intensity")})
    # wandb.log({"SSIM" : wandb.plot.line(table, "I", "SSIM ↑", title="SSIM per Intensity")})
    # wandb.log({"HyperlQA" : wandb.plot.line(table, "I", "HyperlQA ↑", title="HyperlQA per Intensity")})
    # wandb.log({"NIQE" : wandb.plot.line(table, "I", "NIQE ↓", title="NIQE per Intensity")})
    # wandb.log({"BRISQUE" : wandb.plot.line(table, "I", "BRISQUE ↓", title="BRISQUE per Intensity")})
    # wandb.log({"I_sensors" : wandb.plot.line(table, "I", "I_sensors", title="I_sensors per I_norm")})
    # wandb.log({"I_lights" : wandb.plot.line(table, "I", "I_lights", title="I_lights per I_norm")})

def compute_metrics(opt, dsName, testingInOurDataset = True):
    out_dir = opt["root_dir"]
    results = []
    for file in os.listdir(opt["path"]["test_images_dir"]):
        if "gt" not in file and "in.png" not in file and "gamma.png" not in file and "original.png" not in file and "wb.png" not in file:
            pred = PIL.Image.open(opt["path"]["test_images_dir"] + file)

            low = PIL.Image.open(opt["path"]["test_images_dir"] + file[:-4] + "_in.png")
            high = PIL.Image.open(opt["path"]["test_images_dir"] + file[:-4] + "_gt.png")

            results.append([pred, high, low, file])
            file = file.split("-")[0] + "-" + file.split("-")[-1]
            # wandb.log({f"{dsName.capitalize()}-Inference/{file}": wandb.Image(high)})
            # wandb.log({f"{dsName.capitalize()}-Inference/{file}": wandb.Image(low)})
            wandb.log({f"{dsName.capitalize()}-Inference/{file}": wandb.Image(pred)})
    
    plot_metrics(results, opt["test_name"], opt, dsName)

    if testingInOurDataset:
        separate_by_intensity(results, opt, dsName)
        # create_table(opt, dsName)
        create_wandbPlots(opt, dsName)

def evaluate(opt, model, dsName, testingInOurDataset = True):
    opt, model = initialize_test_params(opt, model, dsName = dsName, original = "original" in opt["datasets"]["root"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    generate_test_images(opt, model)
    
    # print("Testing in {}".format("OUR dataset" if testingInOurDataset else "LOL dataset"))
    # print("Grabbing images from {} there are {} images".format(opt["path"]["test_images_dir"], len([file for file in os.listdir(opt["path"]["test_images_dir"]) if "gt" not in file and "in.png" not in file and "gamma.png" not in file and "original.png" not in file and "wb.png" not in file])))
    
    compute_metrics(opt, dsName, testingInOurDataset = testingInOurDataset)

def evaluate_all(opt, model):
    # try: 
        evaluate(opt, model, "nikon")
    # except: 
    #     pass
    
    # try:
    #     evaluate(opt, model, "phone")
    # except:
    #     pass
    
    # try: 
        # evaluate(opt, model, "lol", testingInOurDataset = False)
    # except:
        # pass

if __name__ == "__main__": 
    opt, model = initialize("7-T-0_1", extra = "/znikon_linear_600x400_dif_m/")
    testingInOurDataset = True
    
    print("Testing in {}".format("OUR dataset" if testingInOurDataset else "LOL dataset"))
    # print("Grabbing images from {} there are {} images".format(opt["path"]["test_images_dir"], len([file for file in os.listdir(opt["path"]["test_images_dir"]) if "gt" not in file and "in.png" not in file and "gamma.png" not in file and "original.png" not in file and "wb.png" not in file])))
    
    # if opt["wandb"]["resume"] == "must":
    wandb.init(project="Retinexformer", name=opt["test_name"], config=opt, id="7amz05ah", resume="must")

    # else: 
    # wandb.init(project="Retinexformer", name=opt["test_name"], config=opt)

    evaluate_all(opt, model)

    wandb.finish()
    # /ghome/mpilligua/lowlight/Retinexformer-new/runs/znikon_linear_600x400_dif_m