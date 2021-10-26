import glob
import os
import random

import cv2
import torch
import torchvision
import tqdm
from torchvision import transforms


class Dataset(object):
    def __init__(self, path, cls, mode, aug_mode):
        self.path = path
        self.cls = cls
        self.mode = mode
        self.imgs = list()
        self.aug_mode = aug_mode

        if self.mode == "train":
            self.cutpaste_trans = CutPaste(aug_mode)
            # self.blank_trans = Blank()
            # self.disorder_trans = Disorder()
            folder_path = os.path.join(self.path, self.cls, "train", "good")
            imgpath_list = glob.glob(os.path.join(folder_path, "*.png"))
            pbar = tqdm.tqdm(imgpath_list, desc="loading {} images...".format(self.cls))
            for i in pbar:
                self.imgs.append(cv2.imread(i))
            print("finish loading {:d} images".format(len(self)))

        elif self.mode == "test":
            file_catch = "test" if aug_mode == "test_data_noaug" else "train"
            if file_catch == "train":
                folder_path = os.path.join(self.path, self.cls, "train", "good")
                imgpath_list = glob.glob(os.path.join(folder_path, "*.png"))
                print("loading train images for feature dictionary...")
                for i in imgpath_list:
                    self.imgs.append(cv2.imread(i))
            else:
                self.imgs = glob.glob(os.path.join(self.path, self.cls, "test", "*", "*.png"))

    def __getitem__(self, idx):
        if self.mode == "train":
            img_ori = cv2.resize(self.imgs[idx], (256, 256))
            img_ori = torch.tensor(img_ori[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()

            trans = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                ]
            )

            img_ori = trans(img_ori)
            if self.aug_mode != "3-way":
                img_cutpaste = self.cutpaste_trans(img_ori)
                return img_ori / 255.0, img_cutpaste / 255.0
            else:
                img_cutpaste, scar = self.cutpaste_trans(img_ori)
                return img_ori / 255.0, img_cutpaste / 255.0, scar / 255.0

            # img_blank = self.blank_trans(img_ori)
            # img_disorder = self.disorder_trans(img_ori)

            # return img_ori / 255.0, img_catpaste / 255.0, img_blank / 255.0, img_disorder / 255.0
        else:
            if self.aug_mode == "test_data_noaug":
                img = cv2.imread(self.imgs[idx])
                img = cv2.resize(img, (256, 256))
                img = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
                return img / 255.0, self.imgs[idx].split("/")[-2] == "good"
            else:
                img = cv2.resize(self.imgs[idx], (256, 256))
                img = torch.tensor(img[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
                return img / 255.0

    def __len__(self):
        return len(self.imgs)


class CutPaste(object):
    def __init__(self, aug_mode="cut_paste"):
        self.aug_mode = aug_mode

    def __call__(self, img):
        img = img.clone()
        h, w = img.shape[1], img.shape[2]
        if self.aug_mode == "cut_paste":
            patch_area_r = torch.rand(1) * 0.13 + 0.02  # (0.02,0.15)
            patch_area = patch_area_r.item() * h * w

            ratio_r = 2 * torch.log(torch.tensor(3.0)) * torch.rand(1) - torch.log(torch.tensor(3.0))
            ratio = torch.exp(ratio_r).item()

            patch_h = round((patch_area * ratio) ** 0.5)
            patch_w = round((patch_area / ratio) ** 0.5)

            cut_from_h = int((torch.rand(1) * (h - patch_h + 1)).item())
            cut_from_w = int((torch.rand(1) * (w - patch_w + 1)).item())

            patch = img[:, cut_from_h : cut_from_h + patch_h, cut_from_w : cut_from_w + patch_w].clone()

            color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
            patch = color_jitter(patch)

            new_img = img.clone()
            paste_to_h = int((torch.rand(1) * (h - patch_h + 1)).item())
            paste_to_w = int((torch.rand(1) * (w - patch_w + 1)).item())
            new_img[:, paste_to_h : paste_to_h + patch_h, paste_to_w : paste_to_w + patch_w] = patch

            return new_img  # tensor, cutpaste & color jitter

        elif self.aug_mode == "scar":
            patch_w = torch.randint(2, 17, (1,)).item()
            patch_h = torch.randint(10, 26, (1,)).item()

            cut_from_h = int((torch.rand(1) * (h - patch_h + 1)).item())
            cut_from_w = int((torch.rand(1) * (w - patch_w + 1)).item())

            patch = img[:, cut_from_h : cut_from_h + patch_h, cut_from_w : cut_from_w + patch_w].clone()

            angle = (torch.rand(1) * 90 - 45).item()

            color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
            patch = color_jitter(patch)

            patch = transforms.functional.rotate(patch, angle)
            patch_h = patch.shape[1]
            patch_w = patch.shape[2]

            new_img = img.clone()
            paste_to_h = int((torch.rand(1) * (h - patch_h + 1)).item())
            paste_to_w = int((torch.rand(1) * (w - patch_w + 1)).item())
            new_img[:, paste_to_h : paste_to_h + patch_h, paste_to_w : paste_to_w + patch_w] = patch
            return new_img
        else:
            self.aug_mode = "cut_paste"
            cutpaste = self.__call__(img)
            self.aug_mode = "scar"
            scar = self.__call__(img)
            self.aug_mode = "3-way"
            return cutpaste, scar


class Blank(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.clone()
        h, w = img.shape[1], img.shape[2]

        patch_area_r = torch.rand(1) * 0.13 + 0.02  # (0.02,0.15)
        patch_area = patch_area_r.item() * h * w

        ratio_r = 2 * torch.log(torch.tensor(3.0)) * torch.rand(1) - torch.log(torch.tensor(3.0))
        ratio = torch.exp(ratio_r).item()

        patch_h = round((patch_area * ratio) ** 0.5)
        patch_w = round((patch_area / ratio) ** 0.5)

        cut_from_h = int((torch.rand(1) * (h - patch_h + 1)).item())
        cut_from_w = int((torch.rand(1) * (w - patch_w + 1)).item())

        img[:, cut_from_h : cut_from_h + patch_h, cut_from_w : cut_from_w + patch_w] = 0

        return img


class Disorder:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.clone()
        middel = img[:, 80:160, 80:160].clone()
        img[:, 80:160, 80:160] = 0

        block = 3
        patch = []  # 记录打乱的patch
        row_split = list(torch.chunk(middel, 3, dim=1))  # 沿水平轴切分
        random.shuffle(row_split)  # 沿水平轴shuffle
        for row in row_split:  # 沿垂直轴shuffle
            col_split = list(torch.chunk(row, block, dim=2))
            random.shuffle(col_split)
            shuffle_row_split = torch.cat(col_split, dim=2)
            patch += [shuffle_row_split]
        result = torch.cat(patch, dim=1)

        img[:, 80:160, 80:160] = result
        return img


def collate_func(batch):
    if len(batch[0]) == 3:
        three_way = True
    else:
        three_way = False

    if three_way:
        img, cutpaste, scar = zip(*batch)
        img = torch.stack(img, 0)
        cutpaste = torch.stack(cutpaste, 0)
        scar = torch.stack(scar, 0)
        img_aug = torch.stack((img, cutpaste, scar), 1)
        return img_aug
    else:
        img, cutpaste = zip(*batch)
        img = torch.stack(img, 0)
        cutpaste = torch.stack(cutpaste, 0)
        img_aug = torch.stack((img, cutpaste), 1)
        return img_aug
    # img, aug, blank, disorder = zip(*batch)
    # img = torch.stack(img, 0)
    # aug = torch.stack(aug, 0)
    # blank = torch.stack(blank, 0)
    # disorder = torch.stack(disorder, 0)
    # img_aug = torch.stack((img, aug, blank, disorder), 1)
    # return img_aug  # B 4 C H W


if __name__ == "__main__":
    # for i in range(10):

    #     img = cv2.imread(r'C:\Users\LS\code_python\cutpaste\anomaly_data\bottle\train\good\000.png')
    #     cutpaste = CutPaste()
    #     new_img = cutpaste(img)

    #     cv2.imshow('after', new_img.permute(1,2,0).contiguous().numpy()[:,:,::-1])
    #     cv2.waitKey()
    dataset = Dataset("/home/mxh/anomaly_data", "bottle", "train", "cut_paste")
    # print(len(dataset))
    # data = collate_func([dataset[0], dataset[1], dataset[3], dataset[4]])
    # print(data.shape)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_func,
    )
    for i in dataloader:
        # print(i.shape)
        idx1 = torch.randint(2, (i.shape[0], 1))
        idx2 = 1 - idx1
        idx = torch.cat((idx1, idx2), 1).type(torch.bool)
        # print(idx)
        i = i[idx]
        # print(i.shape)
        label = idx2
        print(label.dtype)
        # for idx, m in enumerate(i):
        #     m = m.permute(1, 2, 0).contiguous().numpy() * 255
        #     cv2.imwrite("{:d}.png".format(idx), m)
        #     print(label[idx])

    # print(dataloader[0].shape)
