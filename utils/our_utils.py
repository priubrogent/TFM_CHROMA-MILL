import os
from glob import glob
import yaml
import shutil

def createDir(name):
    try:
        os.stat(name)
        return 0

    except:
        os.makedirs(name)
        return 1

def build_OURDataset_list_txt(dst_dir, onlyWhite = True):
    # example name : Scene1_a-F_90-SS_125-x_5235-y_3297-B_254 -> Scene1_a-F_90-SS_125-x_5235-y_3297, 254
    img_lr_path = []
    img_hr_path = []
    for name in os.listdir(os.path.join(dst_dir, 'low')):
            
        intensity = name.split("-")[-1].split("_")[-1]
        color = (name.split("-")[3].split("_")[1], name.split("-")[4].split("_")[1])
        scene_name = "-".join(name.split("-")[:-1])
        if onlyWhite:
            if num2col[color] != "White":
                continue
            else: 
                img_lr_path.append(os.path.join('low', name))
                img_hr_path.append(os.path.join('high', scene_name+"-B_254.png"))
        else:
            img_lr_path.append(os.path.join('low', name))
            img_hr_path.append(os.path.join('high', scene_name+"-B_254.png"))

    list_path = os.path.join(dst_dir, 'pair_list.csv')
    with open(list_path, 'w') as f:
        for lr_path, hr_path in zip(img_lr_path, img_hr_path):
            f.write(f"{lr_path},{hr_path}\n")

    return list_path


def watchdogGamma(config):
    config["datasets"]["train"]["gamma_in"] = config["datasets"]["train"].get("gamma_in", 1)
    config["datasets"]["train"]["gamma_train"] = config["datasets"]["train"].get("gamma_train", 1)
    config["datasets"]["train"]["gamma_out"] = config["datasets"]["train"].get("gamma_out", 1)

    config["datasets"]["val"]["gamma_in"] = config["datasets"]["val"].get("gamma_in", 1)
    config["datasets"]["val"]["gamma_train"] = config["datasets"]["val"].get("gamma_train", 1)
    config["datasets"]["val"]["gamma_out"] = config["datasets"]["val"].get("gamma_out", 1)

    config["datasets"]["test"]["gamma_in"] = config["datasets"]["test"].get("gamma_in", 1)
    config["datasets"]["test"]["gamma_train"] = config["datasets"]["test"].get("gamma_train", 1)
    config["datasets"]["test"]["gamma_out"] = config["datasets"]["test"].get("gamma_out", 1)
    
    config["datasets"]["val"]["gamma_in"] = float(eval(str(config["datasets"]["val"]["gamma_in"])))
    config["datasets"]["val"]["gamma_train"] = float(eval(str(config["datasets"]["val"]["gamma_train"])))
    config["datasets"]["val"]["gamma_out"] = float(eval(str(config["datasets"]["val"]["gamma_out"])))

    config["datasets"]["train"]["gamma_in"] = float(eval(str(config["datasets"]["train"]["gamma_in"])))
    config["datasets"]["train"]["gamma_train"] = float(eval(str(config["datasets"]["train"]["gamma_train"])))
    config["datasets"]["train"]["gamma_out"] = float(eval(str(config["datasets"]["train"]["gamma_out"])))

    config["datasets"]["test"]["gamma_in"] = float(eval(str(config["datasets"]["test"]["gamma_in"])))
    config["datasets"]["test"]["gamma_train"] = float(eval(str(config["datasets"]["test"]["gamma_train"])))
    config["datasets"]["test"]["gamma_out"] = float(eval(str(config["datasets"]["test"]["gamma_out"])))
    return config
    
def getCompletePaths(config):
    for split in ["train", "val", "test"]:
        ds = config["datasets"][split]
        root = config["datasets"]["root"]
        if ds.get("data_gt"):
            ds["dataroot_gt"] = root + ds["data_gt"]
        if ds.get("data_lq"):
            ds["dataroot_lq"] = root + ds["data_lq"]
    return config

def setDefColors(config):
    # config["datasets"]["train"]["color_input"] = config["datasets"]["train"].get("color_input", ["White"])
    # config["datasets"]["train"]["color_gt"] = config["datasets"]["train"].get("color_gt", ["White"])
    
    # config["datasets"]["val"]["color_input"] = config["datasets"]["val"].get("color_input", ["White"])
    # config["datasets"]["val"]["color_gt"] = config["datasets"]["val"].get("color_gt", ["White"])

    # config["datasets"]["test"]["color_input"] = config["datasets"]["test"].get("color_input", ["White"])
    # config["datasets"]["test"]["color_gt"] = config["datasets"]["test"].get("color_gt", ["White"])

    config["datasets"]["train"]["color_input"] = ["White"]
    config["datasets"]["train"]["color_gt"] = ["White"]

    config["datasets"]["val"]["color_input"] = ["White"]
    config["datasets"]["val"]["color_gt"] = ["White"]

    config["datasets"]["test"]["color_input"] = ["White"]
    config["datasets"]["test"]["color_gt"] = ["White"]
    return config

def watchdogParams(config):
    config = watchdogGamma(config)
    config["datasets"]["train"]["WB"] = config["datasets"]["train"].get("WB", None)
    config["datasets"]["val"]["WB"] = config["datasets"]["val"].get("WB", None)
    config["datasets"]["test"]["WB"] = config["datasets"]["test"].get("WB", None)

    config["datasets"]["val"]["augmentation"] = config["datasets"]["val"].get("augmentation", None)
    config["datasets"]["test"]["augmentation"] = config["datasets"]["train"].get("augmentation", None)

    config["datasets"]["train"]["i2use"] = config["datasets"]["train"].get("i2use", None)
    config["datasets"]["val"]["i2use"] = config["datasets"]["val"].get("i2use", None)
    config["datasets"]["test"]["i2use"] = config["datasets"]["test"].get("i2use", None)

    config["datasets"]["train"]["normalize_I"] = config["datasets"]["train"].get("normalize_I", None)
    config["datasets"]["val"]["normalize_I"] = config["datasets"]["val"].get("normalize_I", None)
    config["datasets"]["test"]["normalize_I"] = config["datasets"]["test"].get("normalize_I", None)

    config["datasets"]["train"]["Ifrom"] = config["datasets"]["train"].get("Ifrom", "sensors")
    config["datasets"]["val"]["Ifrom"] = config["datasets"]["val"].get("Ifrom", "sensors")
    config["datasets"]["test"]["Ifrom"] = config["datasets"]["test"].get("Ifrom", "sensors")

    # config["train"]["total_epochs"] = config["train"].get("total_epochs", 5000)
    config = getCompletePaths(config)
    config = setDefColors(config)
    return config

def LoadParams_no_prior(name):
    with open("./setup-no_prior/" + name + ".yaml", "r") as f:
        config = yaml.safe_load(f)
        config["test_name"] = name
        config["root_dir"] = "./runs-no_prior/" + name
        config["output_dir"] = "./runs-no_prior/" + name + "/images"
        config["save_weights_dir"] = "./runs-no_prior/" + name + "/weights"
        config["path"]["log"] = "./runs-no_prior/" + name + "/log"

        createDir(config["output_dir"])
        createDir(config["save_weights_dir"])
        createDir(config["path"]["log"])

        if config["path"]["pretrain_network_g"] == None:
            dirs_pth = os.listdir("./runs-no_prior/" + name)

            for dir_pth in dirs_pth:
                if dir_pth[-4:] == ".pth":
                    config["path"]["pretrain_network_g"] = "./runs-no_prior/" + name + "/" + dir_pth
                    
    config = watchdogParams(config)
    return config

def LoadParams_feed_I_with_prior(name):
    with open("./setup-feed_I_with_prior/" + name + ".yaml", "r") as f:
        config = yaml.safe_load(f)
        config["test_name"] = name
        config["root_dir"] = "./runs-feed_I_with_prior/" + name
        config["output_dir"] = "./runs-feed_I_with_prior/" + name + "/images"
        config["save_weights_dir"] = "./runs-feed_I_with_prior/" + name + "/weights"
        config["path"]["log"] = "./runs-feed_I_with_prior/" + name + "/log"

        createDir(config["output_dir"])
        createDir(config["save_weights_dir"])
        createDir(config["path"]["log"])

        if config["path"]["pretrain_network_g"] == None:
            dirs_pth = os.listdir("./runs-feed_I_with_prior/" + name)

            for dir_pth in dirs_pth:
                if dir_pth[-4:] == ".pth":
                    config["path"]["pretrain_network_g"] = "./runs-feed_I_with_prior/" + name + "/" + dir_pth
                    
    config = watchdogParams(config)
    return config


def LoadParams_feed_I(name):
    with open("./setup-feed_I/" + name + ".yaml", "r") as f:
        config = yaml.safe_load(f)
        config["test_name"] = name
        config["root_dir"] = "./runs-feed_I/" + name
        config["output_dir"] = "./runs-feed_I/" + name + "/images"
        config["save_weights_dir"] = "./runs-feed_I/" + name + "/weights"
        config["path"]["log"] = "./runs-feed_I/" + name + "/log"

        createDir(config["output_dir"])
        createDir(config["save_weights_dir"])
        createDir(config["path"]["log"])

        if config["path"]["pretrain_network_g"] == None:
            dirs_pth = os.listdir("./runs-feed_I/" + name)

            for dir_pth in dirs_pth:
                if dir_pth[-4:] == ".pth":
                    config["path"]["pretrain_network_g"] = "./runs-feed_I/" + name + "/" + dir_pth
                    
    config = watchdogParams(config)
    return config


def LoadParams(name, extra = ""):
    with open("./setup/" + extra + name + ".yaml", "r") as f:
        config = yaml.safe_load(f)
        config["test_name"] = name
        config["root_dir"] = "./runs/" + extra  + name
        config["output_dir"] = "./runs/" + extra  + name  + "/images"
        config["save_weights_dir"] = "./runs/" + extra  + name  + "/weights"
        config["path"]["log"] = "./runs/" + extra  + name  + "/log"

        createDir(config["output_dir"])
        createDir(config["save_weights_dir"])
        createDir(config["path"]["log"])

        # if config["path"]["pretrain_network_g"] == None:
        #     dirs_pth = os.listdir("./runs/" + name)

        #     for dir_pth in dirs_pth:
        #         if dir_pth[-4:] == ".pth":
        #             config["path"]["pretrain_network_g"] = "./runs/" + name + "/" + dir_pth
                    
    # config = watchdogParams(config)
    config = getCompletePaths(config)
    # config = setDefColors(config)
    return config

def find_last_weights(opt):
    # iterate over the files in weights_dir and find the one with the highest epoch number
    weights_dir = opt["save_weights_dir"]
    files = os.listdir(weights_dir)
    max_iter = 0
    max_iter_file = None
    for file in files:
        if file[-4:] == ".pth":
            try:
                iterr = int(file.split("_")[-1].split(".")[0])
                if iterr >= max_iter:
                    max_iter = iterr
                    max_iter_file = file
            except:
                continue
    return_path = os.path.join(weights_dir, max_iter_file)

    print("Loading weights from iter: ", max_iter, return_path)
    return return_path, max_iter

def find_best_weights(opt):
    print("Loading best weights from: ", opt["root_dir"] + "/best_psnr.pth")
    return_path = opt["root_dir"] + "/best_psnr.pth"
    return return_path


def get_max_iter(run, path2log):
    path2metrics = path2log + "/metric.csv"
    df = pd.read_csv(path2metrics, comment="i")
    df.columns = ["epoch", "psnr"]
    df["epoch"] = df["epoch"].apply(lambda x: int(x))
    import numpy as _np
    df["psnr"] = df["psnr"].apply(lambda x: eval(x, {"np": _np})["psnr"])
    df.set_index("epoch", inplace=True)

    # remove all the rows with an epoch higher than x
    x = 100000000
    df = df[df.index <= x]

    # print(df)

    return df["psnr"].idxmax()


def ignore_error_lambda(x):
    # lambda x: int(x.split("_")[-1].split(".")[0]) ignoring errors
    try:
        return int(x.split("_")[-1].split(".")[0])
    except:
        return 10000000

import pandas as pd
def find_best_val_weights(opt):
    best_iter = get_max_iter(opt["test_name"], opt["path"]["log"])

    weights_dir = opt["save_weights_dir"]
    files = os.listdir(weights_dir)

    files = sorted(files, key=lambda x: ignore_error_lambda(x))

    # print(files)

    iterr = files[-1]
    best_iter_found = "latest"
    for file in files:
        try:
            iterr = int(file.split("_")[-1].split(".")[0])
            # print(iterr, best_iter, 100000000, best_iter <=iterr, iterr < 100000000)
            if best_iter <= iterr and iterr < 100000000:
                best_iter_found = iterr
                break
        except:
            continue

    return_path = os.path.join(weights_dir, f"net_g_{best_iter_found}.pth")

    print("Loading weights from iter: ", best_iter, return_path, best_iter_found, iterr)
    return return_path, best_iter


if __name__ == "__main__":
    # set working directory /ghome/mpilligua/lowlight/Retinexformer-new

    os.chdir("/ghome/mpilligua/lowlight/Retinexformer-new")

    # L = ['1-T-0_01', '2-T-5', '3-TI-0_1', '4-TI-5', '5-TI-0_01', '6-TI-1', '7-T-0_1', '8-T-1']
    # L = ["9-T-0_1_v2"]
    L = ["Nikon_all_linear", "I-Nikon_all_linear", "T-Nikon_all_linear-m0_1", "T-Nikon_all_linear-m0_01", "T-Nikon_all_linear-m5", "T-Nikon_all_linear", "TI-Nikon_all_linear-m0_1", "TI-Nikon_all_linear-m0_01", "TI-Nikon_all_linear-m5", "TI-Nikon_all_linear"]
    L = ["I-Phone_all_linear2", "I-Phone_all_linear3"]

    for l in L:
        opt = LoadParams(l, extra = "zphone_linear_600x400/")
        path, iter = find_best_val_weights(opt)

        # copy file from path to opt["root_dir"] + "/best_psnr.pth"
        shutil.copy(path, opt["root_dir"] + "/best_iter.pth")

    # delete the file called best_psnr.pth
    # os.remove(opt["root_dir"] + "/best_psnr.pth")

    # print(find_last_weights(opt))
    # print(find_best_weights(opt))