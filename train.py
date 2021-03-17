import sys
import os
import copy
import random
import datetime

import numpy as np
import pickle
import glob
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from pycocotools.coco import COCO

from src.SSAP import SSAP
from src.datasets import COCODataset, make_data_loader
from src.loss import focal_loss, l2_loss, calc_loss
from src.graph_partition import (
    Partition,
    Edge,
    greedy_additive,
    calc_js_div,
    make_ins_seg,
)
from boxx import *
import argparse
import torch.backends.cudnn as cudnn

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex .")


DATA_PATH = "/data/datasets/coco/coco_ssap/"

AFF_R = 5
f = open("./data/t_color.txt", "rb")
t_color = pickle.load(f)


def get_parser():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SSAP instance segmentation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images sent to the network in one step.",
    )
    parser.add_argument(
        "--config",
        default="./src/config.py",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/datasets/coco/",
        help="Path to the directory containing the PASCAL VOC dataset.",
    )
    parser.add_argument(
        "--is-training",
        action="store_true",
        help="Whether to updates the running means and variances during the training.",
    )
    parser.add_argument(
        "--max-epoch", type=int, default=20, help="Number of training steps."
    )
    parser.add_argument(
        "--not-restore-last",
        action="store_true",
        help="Whether to not restore last (FC) layers.",
    )
    parser.add_argument(
        "--log-iter", type=int, default=50, help="Number of training steps."
    )
    # parser.add_argument("--power", type=float, default=POWER,
    #                     help="Decay parameter to compute the learning rate.")
    parser.add_argument(
        "--random-mirror",
        action="store_true",
        help="Whether to randomly mirror the inputs during the training.",
    )
    parser.add_argument(
        "--random-scale",
        action="store_true",
        help="Whether to randomly scale the inputs during the training.",
    )
    # parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
    #                     help="Random seed to have reproducible results.")
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                     help="Where restore model parameters from.")
    # parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
    #                     help="How many images to save.")
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=5000,
        help="Save summaries and checkpoint every often.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="./exp4/",
        help="Where to save snapshots of the model.",
    )
 
    parser.add_argument("--gpu", type=str, default="None", help="choose gpu device.")
    parser.add_argument("--model", type=str, default="None", help="choose model.")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="choose the number of workers."
    )
    return parser


def train():
    parser = get_parser()
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    cudnn.benchmark = True
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        # device = [i for i in range(world_size)]
        args.world_size = world_size
    else:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        # device =  [i for i in range(len(gpus.split(',')))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSAP(n_channels=3, n_classes=len(t_color), aff_r=AFF_R)
    model = model.to(device)

    if distributed:
        model = DistributedDataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[args.local_rank], output_device=args.local_rank,
        #     # this should be removed if we update BatchNorm stats
        #     broadcast_buffers=False, find_unused_parameters=True
        # )
    else:
        model = torch.nn.DataParallel(model)

    aff_calc_weight = [1.5, 0.5]
    aff_weight = 1.0
    l_aff_weight = [1.0, 0.3, 0.1, 0.03, 0.01]

    # Dataset
    img_root = "/data/datasets/coco/train2017/"
    jsp = "/data/datasets/coco/annotations/instances_train2017.json"
    dataset = COCODataset(img_root, jsp, 5)

    train_loader, train_sampler = make_data_loader(args, dataset)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    iter_ = 0
    running_loss = 0.0
    running_loss_seg = 0.0
    running_loss_aff = 0.0
    print("Begin to train model!")
    for epoch in range(args.max_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(train_loader):
            inputs, labels, t_aff = data
            inputs = inputs.to(device)
            labels = labels.type(torch.float32)
            labels = labels.to(device)
            t_aff = t_aff.type(torch.float32)
            t_aff = t_aff.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_seg, loss_aff = calc_loss(
                outputs, labels, t_aff, aff_calc_weight, aff_weight, l_aff_weight
            )
            loss = loss_seg + loss_aff
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            running_loss_seg += loss_seg.data
            running_loss_aff += loss_aff.data

            iter_ += 1

            # log
            if iter_ % args.log_iter == 0:
                with open(args.workdir + "log.txt", mode="a") as f:
                    f.write(
                        "time:{}, epoch:{}, iter:{}, loss:{:.4f}, loss_seg:{:.4f}, loss_aff:{:.4f}\n".format(
                            datetime.datetime.now(),
                            epoch,
                            iter_,
                            loss.data,
                            loss_seg.data,
                            loss_aff.data,
                        )
                    )
                print(
                    "iter:{}, loss:{:.4f}, loss_seg:{:.4f}, loss_aff:{:.4f}".format(
                        iter_, loss.data, loss_seg.data, loss_aff.data
                    )
                )
            # if iter_ % args.save_epoch == 0:
            #     get_affmap(args.workdir,inputs, outputs, t_aff)

            if iter_ % args.save_epoch == 0:
                if (not distributed) or (distributed and args.local_rank == 0):
                    if epoch % args.save_epoch == 0:
                        print("taking checkpoint ...")
                        torch.save(
                            model.state_dict(),
                            args.workdir + "model_" + str(iter_) + ".pth",
                        )

    print("Finished Training")


def get_affmap(save_dir, inputs, outputs, t_aff):
    input_img = inputs[0].cpu().detach().numpy()
    imsave(save_dir + "img.png", np.transpose(input_img, (1, 2, 0)))
    for idx, aff_gt in enumerate(t_aff[0][0].cpu().detach().numpy()):
        imsave(
            save_dir + str(idx) + "aff_gt.png",
            (aff_gt * 255).astype(np.uint8),
        )
        imsave(
            save_dir + str(idx) + "aff_gt.png",
            (aff_gt * 255).astype(np.uint8),
        )

    for idx, pred in enumerate(outputs[9][0].cpu().detach().numpy()):
        imsave(
            save_dir + str(idx) + "_pred.png",
            (pred * 255).astype(np.uint8),
        )
        imsave(
            save_dir + str(idx) + "_pred.png",
            (pred * 255).astype(np.uint8),
        )


if __name__ == "__main__":
    train()
