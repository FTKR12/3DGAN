import torch.utils.data as data
import nibabel as nib
import numpy as np
import glob
import torch
import torchvision.transforms.functional as F
from skimage.transform import resize

from data.patch import make_patches


def make_datadict(opt):
    data_dict = {}
    if opt.phase == 'train':
        id_dirs = glob.glob(opt.dataroot + "/train/*")
        for i, id_dir in enumerate(id_dirs):
            data_dict[i] = {}
            id = id_dir.replace(opt.dataroot + "/train/", "")
            data_dict[i]["id"] = id
            img_paths = glob.glob(id_dir+'/*')
            for img_path in img_paths:
                if "RT1" in img_path: data_dict[i]["RT1"] = img_path
                if "RT2" in img_path: data_dict[i]["RT2"] = img_path
                if "RT3" in img_path: data_dict[i]["RT3"] = img_path
                if "1y" in img_path: data_dict[i]["1y"] = img_path
                if "2y" in img_path: data_dict[i]["2y"] = img_path
                if "LC" in img_path: data_dict[i]["LC"] = img_path
                if "DM" in img_path: data_dict[i]["DM"] = img_path

    elif opt.phase == 'valid':
        id_dirs = glob.glob(opt.dataroot + "/valid/*")
        for i, id_dir in enumerate(id_dirs):
            data_dict[i] = {}
            id = id_dir.replace(opt.dataroot + "/valid/", "")
            data_dict[i]["id"] = id
            img_paths = glob.glob(id_dir+'/*')
            for img_path in img_paths:
                if "RT1" in img_path: data_dict[i]["RT1"] = img_path
                if "RT2" in img_path: data_dict[i]["RT2"] = img_path
                if "RT3" in img_path: data_dict[i]["RT3"] = img_path
                if "1y" in img_path: data_dict[i]["1y"] = img_path
                if "2y" in img_path: data_dict[i]["2y"] = img_path
                if "LC" in img_path: data_dict[i]["LC"] = img_path
                if "DM" in img_path: data_dict[i]["DM"] = img_path

    elif opt.phase == 'test':
        id_dirs = glob.glob(opt.dataroot + "/test/*")
        for i, id_dir in enumerate(id_dirs):
            data_dict[i] = {}
            id = id_dir.replace(opt.dataroot + "/test/", "")
            data_dict[i]["id"] = id
            img_paths = glob.glob(id_dir+'/*')
            for img_path in img_paths:
                if "RT1" in img_path: data_dict[i]["RT1"] = img_path
                if "RT2" in img_path: data_dict[i]["RT2"] = img_path
                if "RT3" in img_path: data_dict[i]["RT3"] = img_path
                if "1y" in img_path: data_dict[i]["1y"] = img_path
                if "2y" in img_path: data_dict[i]["2y"] = img_path
                if "LC" in img_path: data_dict[i]["LC"] = img_path
                if "DM" in img_path: data_dict[i]["DM"] = img_path

    return data_dict

def transform(img, label):
    # only for iseg dataset
    y_img = (img - np.mean(img)) / np.std(img)
    y_label = (label - np.mean(label)) / np.std(label)

    return y_img, y_label

class CustomDataset(data.Dataset):
    def __init__(self, opt):
        self.img_paths = make_datadict(opt)
        self.img_size = len(self.img_paths)
        self.opt = opt

    def __getitem__(self, index):
        img_path = self.img_paths[index]["RT1"]
        id = self.img_paths[index]['id']

        if "1y" in self.img_paths[index].keys():
            label_path = self.img_paths[index]["1y"]
        if "2y" in self.img_paths[index].keys():
            label_path = self.img_paths[index]["2y"]
        if "LC" in self.img_paths[index].keys():
            label_path = self.img_paths[index]["LC"]
        if "DM" in self.img_paths[index].keys():
            label_path = self.img_paths[index]["DM"]
        else:
            label_path = self.img_paths[index]["RT3"]

        high_img = np.array(nib.load(img_path).get_fdata())[:,:,:,np.newaxis].astype(np.float32)  # [H,W,D,1]
        high_label = np.array(nib.load(label_path).get_fdata())[:,:,:,np.newaxis].astype(np.float32)  # [H,W,D,1]
        high_img = resize(high_img, (self.opt.img_width, self.opt.img_height, self.opt.img_depth, self.opt.img_channel))
        high_label = resize(high_label, (self.opt.img_width, self.opt.img_height, self.opt.img_depth, self.opt.img_channel))
        high_img_patches = make_patches(high_img, margin=self.opt.margin, num_patches=self.opt.num_patches)
        high_label_patches = make_patches(high_label, margin=self.opt.margin, num_patches=self.opt.num_patches)
        high_img_patches, high_label_patches = transform(high_img_patches, high_label_patches)

        return {'id': id, 'high_img_patches': high_img_patches, 'high_label_patches': high_label_patches, 'img_path': img_path, 'label_path': label_path}

    def __len__(self):
        return self.img_size


