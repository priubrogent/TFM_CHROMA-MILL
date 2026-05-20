from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize, adjust_gamma

# from data.data_util import paths_from_lmdb
from utils import FileClient, imfrombytes, img2tensor, scandir


class SingleImageDataset(data.Dataset):
    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        self.paths = sorted(list(scandir(self.lq_folder, full_path=True, recursive=True)))
        self.return_I = opt["return_I"]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        
        if self.opt["gamma_in"] != self.opt["gamma_train"]:
            img_lq = adjust_gamma(img_lq, self.opt["gamma_train"]/self.opt["gamma_in"])

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        d2return = {'lq': img_lq, 'lq_path': lq_path}        
        if self.return_I:
            d2return['I'] = 40/1861

        return d2return 

    def __len__(self):
        return len(self.paths)
