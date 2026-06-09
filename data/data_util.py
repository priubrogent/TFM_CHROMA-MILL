import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from data.transforms import mod_crop
from utils import img2tensor, scandir, FileClient, imfrombytes
import os


col2num = {"White": [0.3127, 0.329], "Red": [0.5235,  0.3297], "Green": [0.2821,  0.4727], "Blue":[0.1968, 0.1623]}
def _xy_to_uv(x, y):
    d = -2*x + 12*y + 3
    return 4*x / d, 9*y / d

col2uv = {name: list(_xy_to_uv(*xy)) for name, xy in col2num.items()}

num2col = {(str(int(v[0]*10000)), str(int(v[1]*10000))): k for k, v in col2num.items()}
getIntensityV = {"low": [0, 80, 113], "mid": [139, 161, 180, 197], "high": [227, 213, 241], "all": [0, 80, 113, 139, 161, 180, 197, 227, 213, 241]}


def col2string(str_color):
    str_color = str_color.lower().capitalize()
    num_color = col2num[str_color]
    # print(str_color, num_color, (num_color[0])
    return "x_" + str(int(float(num_color[0])*10000)) + "-y_" + str(int(float(num_color[1])*10000))

def string2col(str_color):
    num_color = str_color.split("-")
    return num2col[(num_color[0].split("_")[-1], num_color[1].split("_")[-1])]

def get_xColor_elements(list_elements, color2search):
    # list_elements: list of elements to search
    # color2search: color to search {"White", "Red", "Green", "Blue"}
    color2search = color2search.lower().capitalize()
    color2search = col2string(color2search)

    list_elements = [element for element in list_elements if color2search in element]
    return list_elements

def paired_paths_from_folder_custom(folders, keys, opt):
    if 'filename_tmpl' in opt:
            filename_tmpl = opt['filename_tmpl']
    else:
        filename_tmpl = '{}'
    i2use = opt['i2use']
    color_input = opt['color_input']
    color_gt = opt.get('color_gt', ['White'])
    b_gt = opt.get('b_gt', [254])
    # color_input = ["White"]
    # color_gt = ["White"]

    input_folder, gt_folder = folders

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))

    i2use = getIntensityV[i2use.lower()]

    paths = []
    for name in input_paths:
        if "SS_125" in name: # TODO: Arreglar per no haver de treure les problematiques
            continue
        scene_name = "-".join(name.split("-")[:-1])
        color_str = "-".join(name.split("-")[-3:-1])

        if i2use is not None:
            i = int(name.split("-")[-1].split("_")[-1].split(".")[0])
            if i not in i2use:
                continue

        if color_input == 'all':
            try:
                colors_to_try = [string2col(color_str)]
            except KeyError:
                continue
        else:
            colors_to_try = color_input

        for color in colors_to_try:
            color = color.lower().capitalize()
            if col2string(color) == color_str:
                input_path = os.path.join(input_folder, name)

                gt_color = color if color in color_gt else color_gt[0]
                new_color = col2string(gt_color.lower().capitalize())
                scene_name_gt = scene_name.replace(color_str, new_color)
                llie_gt_path = os.path.join(gt_folder, scene_name_gt + "-B_254.png")
                wb_gt_path = os.path.join(input_folder, scene_name_gt + f"-B_{i}.png")
                if len(b_gt) == 1 and b_gt[0] == 254:
                    gt_path = llie_gt_path
                else:
                    gt_path = wb_gt_path

                paths.append(dict([('lq_path', input_path), ('gt_path', gt_path), ('wb_gt_path', wb_gt_path), ('llie_gt_path', llie_gt_path), ('color', color), ('intensity', i), ('chroma', col2num[color])]))
    return paths



def paired_paths_from_RAISE(folders, keys, opt):
    if 'filename_tmpl' in opt:
            filename_tmpl = opt['filename_tmpl']
    else:
        filename_tmpl = '{}'

    input_folder = folders
    input_paths = list(scandir(input_folder))

    paths = []
    for name in input_paths:
        scene_name = "-".join(name.split("_")[:-1])
        i = int(name.split("_")[-1].split(".")[0])
        input_path = os.path.join(input_folder, name)
        gt_path = os.path.join(input_folder, scene_name+"_10.png")
        paths.append(dict([('lq_path', input_path), ('gt_path', gt_path), ('intensity', i)]))
    return paths



def paired_paths_from_folder_custom_no_color(folders, keys, opt):
    i2use = opt['i2use']
    
    input_folder, gt_folder = folders

    input_paths = list(scandir(input_folder))

    i2use = getIntensityV[i2use.lower()]

    paths = []
    for name in input_paths:
        scene_name = "-".join(name.split("-")[:-1])

        if i2use is not None:
            i = int(name.split("-")[-1].split("_")[-1].split(".")[0])
            if i not in i2use:
                continue

        input_path = os.path.join(input_folder, name)
        
        gt_path = os.path.join(gt_folder, scene_name+"-B_254.png")
            
        paths.append(dict([('lq_path', input_path), ('gt_path', gt_path), ('color', 'White'), ('intensity', i)]))
    return paths

def paired_paths_from_folder(folders, keys, opt):
    if 'filename_tmpl' in opt:
            filename_tmpl = opt['filename_tmpl']
    else:
        filename_tmpl = '{}'
    
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))

    
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        # if "Scene1_a" in gt_paths[idx]:
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                        f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                (f'{gt_key}_path', gt_path)]))
    return paths


import random
def collate_fn(batch, transform=None, phase='train'):
    # batch is a list of dicts
    # out is a dict with keys: lq, gt, lq_path, gt_path
    out = {}
    if "anchor" in batch[0].keys():
        # print(batch[0]["anchor"]["lq"].shape)
        # exit(0)
        w = batch[0]["anchor"]["lq"].shape[1]
        h = batch[0]["anchor"]["lq"].shape[2]

        x = random.randint(0, w - 128)
        y = random.randint(0, h - 128)
        for name in ["anchor", "positive", "negative"]:
            if out.get(name) is None:
                out[name] = {}
                # out[name] = {"gt": [], "lq": [], "lq_path": [], "gt_path": [], "I": [], "label": []}
            for key in batch[0][name].keys():
                if isinstance(batch[0][name][key], str):
                    out[name][key] = [b[name][key] for b in batch]
                elif isinstance(batch[0][name][key], int) or isinstance(batch[0][name][key], float):
                    out[name][key] = torch.tensor([b[name][key] for b in batch], dtype=torch.float32)
                else:
                    temp = torch.stack([b[name][key] for b in batch], dim=0)
                    out[name][key] = temp
                
            if transform is not None and phase == 'train':
                # NEW CURRICULUM: triplet sub-dicts now have gt_wb/gt_combined instead of gt
                sub = out[name]
                if 'gt_wb' in sub and 'gt_combined' in sub:
                    B = sub['lq'].shape[0]
                    lq_doubled = torch.cat([sub['lq'], sub['lq']], dim=0)
                    gt_all     = torch.cat([sub['gt_wb'], sub['gt_combined']], dim=0)
                    if name == "anchor" or name == "positive":
                        lq_out, gt_out = transform(lq_doubled, gt_all, x, y)
                    else:
                        lq_out, gt_out = transform(lq_doubled, gt_all)
                    sub['lq']          = lq_out[:B]
                    sub['gt_wb']       = gt_out[:B]
                    sub['gt_combined'] = gt_out[B:]
                else:
                    if name == "anchor" or name == "positive":
                        sub["lq"], sub["gt"] = transform(sub["lq"], sub["gt"], x, y)
                    else:
                        sub["lq"], sub["gt"] = transform(sub["lq"], sub["gt"])
                # END NEW CURRICULUM
    else: 
        for key in batch[0].keys():
            if isinstance(batch[0][key], str):
                out[key] = [b[key] for b in batch]
            elif isinstance(batch[0][key], int) or isinstance(batch[0][key], float):
                out[key] = torch.tensor([b[key] for b in batch], dtype=torch.float32)
            else:
                out[key] = torch.stack([b[key] for b in batch], dim=0)

        if transform is not None and phase == 'train':
            # NEW CURRICULUM: gt_wb and gt_combined must receive the identical random
            # crop and augmentation as lq. We achieve this by doubling the batch so
            # RandomBatchCrop draws one (x, y, mode) that is applied to all images.
            if 'gt_wb' in out and 'gt_combined' in out:
                B = out['lq'].shape[0]
                lq_doubled = torch.cat([out['lq'], out['lq']], dim=0)
                gt_all     = torch.cat([out['gt_wb'], out['gt_combined']], dim=0)
                lq_out, gt_out = transform(lq_doubled, gt_all)
                out['lq']          = lq_out[:B]
                out['gt_wb']       = gt_out[:B]
                out['gt_combined'] = gt_out[B:]
            else:
                out["lq"], out["gt"] = transform(out["lq"], out["gt"])
            # END NEW CURRICULUM
        # print(out["lq_path"])

    return out

