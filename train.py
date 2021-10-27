import argparse
import math
import os

import cv2
import numpy as np
import torch

# import torch.distributed as dist
import torch.nn as nn

# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

from cutpaste import CPmodel, data
from eval import eval

model_names = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
parser.add_argument("--data", default="/home/mxh/anomaly_data/")
parser.add_argument("-a", default="resnet18", choices=model_names, dest="model_name")
parser.add_argument("--class", default="bottle", dest="cls")
parser.add_argument("--aug-mode", default="cut_paste", choices=["cut_paste", "scar", "3-way"])
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--pretrained", default=False, action="store_true")
parser.add_argument("--epochs", default=2, type=int)
parser.add_argument("--lr", default=0.03, type=int)
parser.add_argument("--test-interval", default=10, type=int)
args = parser.parse_args()
print(args)


def train(args):

    # dist.init_process_group(backend="nccl")
    writer = SummaryWriter("log/{}".format(args.cls))

    dataset = data.Dataset(args.data, args.cls, "train", args.aug_mode)
    # data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    model = CPmodel.Model(args.model_name, args.aug_mode, args.pretrained)
    model.to("cuda")
    # model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=3e-5)
    loss_func1 = nn.CrossEntropyLoss().to("cuda")
    loss_func2 = nn.CrossEntropyLoss().to("cuda")
    loss_func3 = nn.MSELoss().to("cuda")
    result_list = []

    for epoch in range(args.epochs):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=data.collate_func,
            drop_last=True,
        )
        # dataloader.sampler.set_epoch(epoch)
        total_loss = 0

        for input in dataloader:  # B 4 C H W    # img, paste, blank, disorder
            batchsize = input.shape[0]
            # idx_paste = torch.randint(2, (batchsize // 3, 1))
            # idx_paste = torch.randint(2, (batchsize, 1))
            # label_paste = 1 - idx_paste
            # idx_paste = torch.cat((idx_paste, label_paste), 1).type(torch.bool)
            # cutpaste = input[: batchsize // 3, :2, ...][idx_paste]  # B/3 C H W
            # cutpaste = input[:batchsize, :2, ...][idx_paste]  # B/3 C H W

            # ori = input[-batchsize // 3 :, 0, ...]  # B/3 C H W
            # ori = input[:batchsize, 0, ...]  # B/3 C H W
            # ori_std = input[batchsize // 3 : -batchsize // 3, 0, ...]  # B/3 C H W
            # blank = input[-batchsize // 3 :, 2, ...]  # B/3 C H W

            # idx_disorder = torch.randint(2, (batchsize // 3, 1))
            # label_disorder = 1 - idx_disorder
            # idx_disorder = torch.cat((idx_disorder, label_disorder), 1).type(torch.bool)
            # disorder = input[batchsize // 3 : -batchsize // 3, [0, 3], ...][idx_disorder]  # B/3 C H W

            # input = torch.cat((cutpaste, disorder, blank), 0)  # B C H W
            num_cls = input.shape[1]
            # label_paste = torch.randint(num_cls, (batchsize,))
            # label = label_paste.tolist()
            # idx = torch.zeros((batchsize, num_cls))
            # for i in range(batchsize):
            #     idx[i][label[i]] = 1
            # idx = idx.type(torch.bool)
            # cutpaste = input[idx]
            stacking = torch.cat([input[:, i, ...] for i in range(num_cls)], 0)
            y = torch.arange(num_cls)
            y = y.repeat_interleave(batchsize)
            m = torch.randperm(num_cls * batchsize)
            input = stacking[m].to("cuda")
            label = y[m].to("cuda")

            # input = torch.cat((cutpaste, ori_std, blank), 0)  # B C H W
            # for ind, i in enumerate(input):  # test code
            #     img = i.permute(1, 2, 0).contiguous().cpu().numpy() * 255
            #     cv2.imwrite("{}.png".format(ind), img)
            #     print(label[ind])
            # if ind < 16:
            #     print(label_paste[ind], end=" ")
            # elif ind < 32:
            #     print(label_disorder[ind - 16], end=" ")
            # else:
            #     pass
            # exit()
            # input = input.to("cuda")
            # ori = ori.to("cuda")

            emb, paste = model(input)  # Bx2, Bx2, Bx3x256x256
            # embed, paste, disorder, blank = model(input)  # Bx2, Bx2, Bx3x256x256

            std = torch.std(emb[label == 0], dim=0)

            # label_paste = label_paste.view(-1).to("cuda")
            # label_disorder = label_disorder.view(-1).to("cuda")

            # loss1 = loss_func1(paste[: batchsize // 3], label_paste)
            loss1 = loss_func1(paste, label)
            # loss2 = loss_func2(disorder[batchsize // 3 : -batchsize // 3], label_disorder)
            # loss3 = loss_func3(ori.reshape(-1), blank[-batchsize // 3 :, ...].reshape(-1))
            loss2 = torch.mean(std)
            # loss = loss1 + loss2 + loss3 + loss4
            loss = loss1 + loss2
            loss.backward()
            total_loss += loss
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 30 == 0:
            print("epoch: {:d}/{:d}  loss: {:4f}".format(epoch, args.epochs, total_loss / len(dataloader)))
            writer.add_scalar("loss", total_loss / len(dataloader), epoch)
            # writer.add_scalar("loss1", loss1, epoch)
            # writer.add_scalar("loss2", loss2, epoch)
            # writer.add_scalar("loss3", loss3, epoch)
            # writer.add_scalar("loss4", loss4, epoch)

        if epoch != 0 and (epoch % args.test_interval == 0 or epoch == args.epochs - 1):
            roc_auc = eval(args.data, args.cls, args.batch_size, gpu=0, model=model)
            writer.add_scalar("roc_auc", roc_auc, epoch)
            print("epoch {} roc_auc {:3f}".format(epoch, roc_auc))
            result_list.append(roc_auc)
            model.train()

            # if epoch % args.test_interval == 0:
            # roc_auc = eval(args.data, args.cls, args.batch_size, gpu=0, model=model)
            # writer.add_scalar('roc_auc', roc_auc, total_iter)
            # model.train()
        adjust_learning_rate(optimizer, epoch)

    info_dict = {
        "model_name": args.model_name,
        "class": args.cls,
        "aug_mode": args.aug_mode,
        "batch_size": args.batch_size,
        "pretrained": args.pretrained,
        "check_point": model.state_dict(),
    }
    torch.save(info_dict, "{}.ckpt".format(args.cls))
    return result_list


def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    # if epoch < 2 * args.epochs / 3:
    #     lr = args.lr
    # else:
    #     lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    times_trial = 1
    if args.cls == "all":
        classes = [
            "carpet",
            "grid",
            "leather",
            "tile",
            "wood",
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "metal_nut",
            "pill",
            "screw",
            "toothbrush",
            "transistor",
            "zipper",
        ]
    else:
        classes = args.cls.split(" ")
    for cls in classes:
        final_rst = []
        for i in range(times_trial):
            args.cls = cls
            result_list = train(args)
            final_rst.append(result_list[-1])
        final_rst = np.array(final_rst)
        with open("result.txt", "a") as f:
            f.write(
                "result for class {} --- mean: {:.3f}   std: {:.3f}\n".format(
                    cls, np.mean(final_rst), np.std(final_rst)
                )
            )
