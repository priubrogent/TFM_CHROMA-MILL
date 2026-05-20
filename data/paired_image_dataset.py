from torch.utils import data as data
from torchvision.transforms.functional import normalize, adjust_gamma
from torchvision.utils import save_image
from torchvision import transforms
import sys
sys.path.append("/ghome/mpilligua/lowlight/Models/Retinexformer-new/basicsr")

from data.data_util import *
from data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation, wb
from utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from pdb import set_trace as stx
from PIL import Image
# from deep_wb import deep_wb_single_task
# from deep_wb.deep_wb_model import deepWBNet
# from deep_wb.deepWB import deep_wb

I2sensor = {'Red': {254: 1002.4110000000001, 241: 902.8583, 227: 801.4689, 213: 706.1163, 197: 604.5339, 180: 505.242, 161: 404.8023, 139: 302.3945, 113: 200.58630000000002, 80: 101.352, 0: 49}, 'Green': {254: 1926.4376, 241: 1734.1636999999998, 227: 1538.4017000000001, 213: 1354.3604999999998, 197: 1158.3797, 180: 966.924, 161: 773.3957, 139: 576.2801, 113: 380.6405, 80: 190.54399999999998, 0: 49}, 'Blue': {254: 193.59879999999998, 241: 173.97789999999998, 227: 154.0195, 213: 135.2763, 197: 115.34349999999999, 180: 95.904, 161: 76.2979, 139: 56.3923, 113: 36.7363, 80: 17.823999999999998, 0: 49}, 'White': {254: 1861.312, 241: 1675.5043, 227: 1486.3278999999998, 213: 1308.4803, 197: 1119.0979, 180: 934.0919999999999, 161: 747.0883, 139: 556.6255, 113: 367.60029999999995, 80: 183.95199999999997, 0: 49}}

class Dataset_PairedImage_Custom(data.Dataset):
    def __init__(self, opt, ourDS=True):
        super(Dataset_PairedImage_Custom, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.ourDS = ourDS
        if ourDS:
            self.paths = paired_paths_from_folder_custom([self.lq_folder, self.gt_folder], ['lq', 'gt'], opt) # self.filename_tmpl, i2use = opt['i2use'])
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], opt) # self.filename_tmpl, i2use = opt['i2use']

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        self.return_I = opt['return_I']
        self.return_chroma = opt.get('return_chroma', False)

    def get_I(self, data):
        I = data['intensity']
        col = data['color']

        if self.opt["Ifrom"] == "sensor":
            I = I2sensor[col][I]/max(I2sensor[col].values())
        
        else: 
            I /= 254
        
        return I
    def get_chroma(self, data):
        col = data['color']
        chroma = col2num[col]
        return torch.tensor(chroma, dtype=torch.float32)
    
    def prepare_item(self, data):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = data['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = data['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # use pytorch transform to resize the image to 6000x4000 at max
        if img_gt.shape[0] > 6000 or img_gt.shape[1] > 4000:
            img_gt = cv2.resize(img_gt, (3000, 2000), interpolation=cv2.INTER_CUBIC) # INTER_NEAREST)
            img_lq = cv2.resize(img_lq, (3000, 2000), interpolation=cv2.INTER_CUBIC) # INTER_NEAREST)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        if self.opt["gamma_in"] != self.opt["gamma_train"]:
            img_lq = adjust_gamma(img_lq, self.opt["gamma_train"]/self.opt["gamma_in"])
            img_gt = adjust_gamma(img_gt, self.opt["gamma_train"]/self.opt["gamma_in"])

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # print("img_lq.shape:", img_lq.shape, "img_gt.shape:", img_gt.shape)

        d2return = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        if self.return_I:
            if self.ourDS:
                I = self.get_I(data)
                d2return['I'] = I
            else:
                d2return['I'] = 40/1861
        
        if self.return_chroma:
            if self.ourDS:
                chroma = self.get_chroma(data)
                d2return['chroma_gt'] = chroma
            else:
                d2return['chroma_gt'] = 0.5
        return d2return

    def __getitem__(self, index):
        index = index % len(self.paths)
        data = self.paths[index]
        return self.prepare_item(data)

    def __len__(self):
        return len(self.paths)

class Dataset_PairedImage_Custom_Correct(data.Dataset):
    def __init__(self, opt, ourDS=True):
        super(Dataset_PairedImage_Custom_Correct, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.ourDS = ourDS
        if ourDS:
            self.paths = paired_paths_from_folder_custom([self.lq_folder, self.gt_folder], ['lq', 'gt'], opt) # self.filename_tmpl, i2use = opt['i2use'])
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], opt) # self.filename_tmpl, i2use = opt['i2use']

        # if self.opt['phase'] == 'train':
            # self.geometric_augs = opt['geometric_augs']

        self.return_I = opt['return_I']
        self.return_chroma = opt.get('return_chroma', False)


    def get_I(self, data):
        I = data['intensity']
        col = data['color']

        if self.opt["Ifrom"] == "sensor":
            I = I2sensor[col][I]/max(I2sensor[col].values())
        
        else: 
            I /= 254
        
        return I
    
    def get_chroma(self, data):
        col = data['color']
        chroma = col2num[col]
        return torch.tensor(chroma, dtype=torch.float32)
    
    def prepare_item(self, data):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = data['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = data['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # use pytorch transform to resize the image to 6000x4000 at max
        if img_gt.shape[0] > 6000 or img_gt.shape[1] > 4000:
            img_gt = cv2.resize(img_gt, (3000, 2000), interpolation=cv2.INTER_CUBIC) # INTER_NEAREST)
            img_lq = cv2.resize(img_lq, (3000, 2000), interpolation=cv2.INTER_CUBIC) # INTER_NEAREST)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        if self.opt["gamma_in"] != self.opt["gamma_train"]:
            img_lq = adjust_gamma(img_lq, 1/2.2)
            img_gt = adjust_gamma(img_gt, 1/2.2)


        # print("max(img_lq):", img_lq.max(), "min(img_lq):", img_lq.min())
        # print("max(img_gt):", img_gt.max(), "min(img_gt):", img_gt.min())
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)



        # print("img_lq.shape:", img_lq.shape, "img_gt.shape:", img_gt.shape)

        d2return = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        
        if self.return_I:
            if self.ourDS:
                I = self.get_I(data)
                d2return['I'] = I
            else:
                d2return['I'] = 40/1861
        
        if self.return_chroma:
            if self.ourDS:
                chroma = self.get_chroma(data)
                d2return['chroma_gt'] = chroma
            else:
                d2return['chroma_gt'] = 0.5
        return d2return


        return d2return

    def __getitem__(self, index):
        index = index % len(self.paths)
        data = self.paths[index]
        return self.prepare_item(data)

    def __len__(self):
        return len(self.paths)



class Triplet(Dataset_PairedImage_Custom_Correct):
    def __init__(self, opt):
        super(Triplet, self).__init__(opt)

        self.groupByScene()

    def groupByScene(self):
        # self.paths -> list of dicts with keys: lq_path, gt_path, color
        # out -> self.dicScenes -> dict with keys: scene_name, list of dicts with keys: lq_path, gt_path, color
        self.dicScenes = {}

        for path in self.paths:
            scene_name = "-".join(path['lq_path'].split('/')[-1].split('-')[:3])
            if scene_name not in self.dicScenes:
                self.dicScenes[scene_name] = []
            self.dicScenes[scene_name].append(path)

    def get_triplet(self, scene, ind_I):
        element = self.dicScenes[scene][ind_I]
        triplet = {}
        triplet["anchor"] =  element
        # get a random index from the list of images that have the same scene and dif I
        ind = random.randint(0, len(self.dicScenes[scene])-1)
        while ind == ind_I: ind = random.randint(0, len(self.dicScenes[scene])-1)
        triplet["positive"] = self.dicScenes[scene][ind]

        # get a random index from the list of images that have the dif scene 
        scene_ = random.choice(list(self.dicScenes.keys()))
        while scene_ == scene: scene_ = random.choice(list(self.dicScenes.keys()))
        ind_ = random.randint(0, len(self.dicScenes[scene_])-1)
        triplet["negative"] = self.dicScenes[scene_][ind_]

        return triplet

    def __getitem__(self, index):
        scene, ind = self.getInfo(self.paths[index])

        triplet = self.get_triplet(scene, ind)

        anchor = self.prepare_item(triplet["anchor"])
        positive = self.prepare_item(triplet["positive"])
        negative = self.prepare_item(triplet["negative"])

        return {"anchor": anchor, "positive": positive, "negative": negative}

    def getInfo(self, data):
        scene = "-".join(data['lq_path'].split('/')[-1].split('-')[:3])
        ind = self.dicScenes[scene].index(data)

        return scene, ind



class TripletMining(Dataset_PairedImage_Custom_Correct):
    def __init__(self, opt):
        super(TripletMining, self).__init__(opt)

        self.groupByScene()

    def groupByScene(self):
        # self.paths -> list of dicts with keys: lq_path, gt_path, color
        # out -> self.dicScenes -> dict with keys: scene_name, list of dicts with keys: lq_path, gt_path, color
        self.dicScenes = {}

        for path in self.paths:
            scene_name = "-".join(path['lq_path'].split('/')[-1].split('-')[:3])
            if scene_name not in self.dicScenes:
                self.dicScenes[scene_name] = []
            self.dicScenes[scene_name].append(path)

        self.orderScenes = list(self.dicScenes.keys())
        # print("self.orderScenes:", self.orderScenes)

    def __getitem__(self, index):
        scene, ind, label = self.getInfo(self.paths[index])

        data = self.prepare_item(self.paths[index])
        data["label"] = label

        return data 

    def getInfo(self, data):
        scene = "-".join(data['lq_path'].split('/')[-1].split('-')[:3])
        ind = self.dicScenes[scene].index(data)
        label = self.orderScenes.index(scene)

        return scene, ind, label



class Dataset_PairedImage(Dataset_PairedImage_Custom):
    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__(opt, ourDS=False)



class RAISE_dataset(data.Dataset):
    def __init__(self, opt):
        super(RAISE_dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        scenes = opt['scenes'] if 'scenes' in opt else None
        print("using scenes:", scenes)
        
        self.lq_folder = opt['dataroot_lq']
        if scenes is None:
            self.paths = paired_paths_from_RAISE(self.lq_folder, 'lq', opt) # self.filename_tmpl, i2use = opt['i2use'])
        
        else:
            self.paths = []
            for scene in scenes:
                for i in range(11):
                    self.paths.append({'lq_path': f"{self.lq_folder}/{scene}_{i:02d}.png", 
                                        'gt_path': f"{self.lq_folder}/{scene}_10.png", 
                                        'intensity': i})
        
        self.return_I = opt['return_I']

    def get_I(self, data):
        I = data['intensity'] / 10
        # I = ((data['intensity'] / 10)**(1/2.2)) * 254
        return I

    def prepare_item(self, data):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = data['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = data['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        # try:
        img_lq = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path))

        
        # Rotate the image if the vertical shape is bigger than the horizontal
        # if img_gt.shape[0] > img_gt.shape[1]:
        #     img_gt = np.rot90(img_gt)
        #     img_lq = np.rot90(img_lq)
        
        # resize to 600x400
        # img_gt = cv2.resize(img_gt, (400, 600), interpolation=cv2.INTER_CUBIC)
        # img_lq = cv2.resize(img_lq, (400, 600), interpolation=cv2.INTER_CUBIC)
        
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        I = self.get_I(data)
        d2return = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'I': I}
        return d2return

    def __getitem__(self, index):
        index = index % len(self.paths)
        data = self.paths[index]
        return self.prepare_item(data)

    def __len__(self):
        return len(self.paths)


class RAISE_dataset_some(data.Dataset):
    def __init__(self, opt):
        super(RAISE_dataset_some, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.lq_folder = opt['dataroot_lq']
        self.paths = []
        
        scenes = opt['scenes'] if 'scenes' in opt else None
        print("using scenes:", scenes)
        
        for scene in scenes:
            for i in range(11):
                self.paths.append({'lq_path': f"{self.lq_folder}/{scene}_{i:02d}.png", 
                                    'gt_path': f"{self.lq_folder}/{scene}_10.png", 
                                    'intensity': i/10})
        
        self.return_I = opt['return_I']

    def get_I(self, data):
        I = data['intensity'] / 10
        return I

    def prepare_item(self, data):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = data['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = data['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        # try:
        img_lq = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path))

        
        # Rotate the image if the vertical shape is bigger than the horizontal
        # if img_gt.shape[0] > img_gt.shape[1]:
        #     img_gt = np.rot90(img_gt)
        #     img_lq = np.rot90(img_lq)
        
        # resize to 600x400
        # img_gt = cv2.resize(img_gt, (400, 600), interpolation=cv2.INTER_CUBIC)
        # img_lq = cv2.resize(img_lq, (400, 600), interpolation=cv2.INTER_CUBIC)
        
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        I = self.get_I(data)
        d2return = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'I': I}
        return d2return

    def __getitem__(self, index):
        index = index % len(self.paths)
        data = self.paths[index]
        return self.prepare_item(data)

    def __len__(self):
        return len(self.paths)





if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    
    from data.transforms import RandomBatchCrop
    
    # test Triplet
    opt = {
        'dataroot_gt': '/ghome/mpilligua/lowlight/OURdataset_linear/nikon/train/high',
        'dataroot_lq': '/ghome/mpilligua/lowlight/Datasets/RAISE/train',
        'io_backend': {'type': 'disk'},
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'phase': 'train',
        'scale': 1,
        'geometric_augs': True,
        'return_I': False,
        'gamma_in': 1,
        'gamma_train': 1,
        'i2use': "all",
        'color_input': 'RGB',
        'gt_size': 400,
    }

    dataset = RAISE_dataset_some(opt)
    print("len(dataset):", len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn= lambda batch: collate_fn(batch, RandomBatchCrop(128), phase='train'))

    for i, data in enumerate(dataloader):
        print(data)
        break
        # if i == 1: break


    # test TripletMining
    # opt = {
    #     'dataroot_gt': '/ghome/mpilligua/lowlight/OURdataset_linear/nikon/train/high',