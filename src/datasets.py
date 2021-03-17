import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
from pycocotools.coco import COCO
import pickle
from boxx import *


class COCODataset(torch.utils.data.Dataset):
    """
    COCODataset

    """
    def __init__(self, img_dir, json_path,aff_r):
        """__init__

            Args:
                img_dir: img 路径
                json_path: COCO annotation路径

        """
        self.img_dir = img_dir
        self.jsp = json_path
        self.img_size = 512
        self.coco = COCO(json_path)

        self.img_file = []
        for json_img in self.coco.dataset["images"]:
            self.img_file.append(json_img["file_name"])
            
        self.aff_r = aff_r #affinity 
        self.aff_resolution = 5
        mean = [0.477, 0.451, 0.411]
        std = [0.284, 0.280, 0.292]
        self.transform = transforms.ToTensor()
        self.transform_img = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=mean,
                                                       std=std)])

        f = open('./data/t_color.txt',"rb")
        self.t_color = pickle.load(f)
        f = open('./data/t_class_name.txt',"rb")
        self.t_color_name = pickle.load(f)
        self.labels = self.t_color

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        """__getitem__

            Args:
                idx (int):  遍历

            Returns:
                Tensor: img (n_batch, ch, height, width)
                Tensor: sem_seg (n_batch, class数, height, width)
                Tensor: aff_map (n_batch, aff_r, aff_r**2, height, width)

        """
        #read img_file
        img_name = self.img_file[idx]
        img = np.array(Image.open(self.img_dir+img_name))
        width, height = img.shape[0],img.shape[1]
        if len(img.shape)<3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_seg = np.zeros((width,height,3), dtype=int)
        img_ins = np.zeros((width,height,3), dtype=int)

        #read ins
        id_ = int(img_name.split("/")[-1][-10:-4])
        pre_color = []
        for i in range(len(self.t_color_name)-1):
            catIds = self.coco.getCatIds(catNms=[self.t_color_name[i]])
            annIds = self.coco.getAnnIds(imgIds=id_, catIds=catIds, iscrowd=False)
            if not annIds:
                continue
            anns = self.coco.loadAnns(annIds)
            for ann in anns:
                while(True):
                    color = np.random.randint(1, 255, 3)
                    if not [j for j in range(len(pre_color))
                            if np.sum(pre_color[j] == color) == 3]:
                        pre_color.append(color)
                        break
                mask = self.coco.annToMask(ann)
                mask = np.array(mask, dtype=int)
                mask_seg = mask[..., None] * self.t_color[i]
                mask_ins = mask[..., None] * color

                img_seg = np.where(img_seg == 0, mask_seg, img_seg)
                img_ins = np.where(img_ins == 0, mask_ins, img_ins)
        
        img = Image.fromarray(img)
        img_seg = Image.fromarray(np.uint8(img_seg))
        img_ins = Image.fromarray(np.uint8(img_ins))

        #resize
        w, h = self.get_size((width,height))

        img = img.resize((w,h))
        img_seg = img_seg.resize((w,h))
        img_ins = img_ins.resize((w,h))

        #crop
        crop_size = self.img_size
        x = np.random.randint((w - crop_size)+1)
        y = np.random.randint((h - crop_size)+1)

        img = img.crop((x, y, x+crop_size, y+crop_size))
        img_seg = img_seg.crop((x, y, x+crop_size, y+crop_size))
        img_ins = img_ins.crop((x, y, x+crop_size, y+crop_size))

        #获得了img,seg,ins, 继续获取aff_gt
        sem_seg = np.array(img_seg)
        img_ins = np.array(img_ins)
        img_t_cls = np.zeros((img.size[0], img.size[1], len(self.labels)))
        # semantic标签
        for i in range(len(self.labels)):
            img_t_cls[:, :, i] = np.where((sem_seg[:, :, 0] == self.labels[i][0])
                           & (sem_seg[:, :, 1] == self.labels[i][1])
                           & (sem_seg[:, :, 2] == self.labels[i][2]), 1, 0)

        out_data = torch.zeros((3,self.img_size, self.img_size))
        out_t = torch.zeros((len(self.labels),self.img_size, self.img_size))

        aff_map = self.Affinity_generator_new(img_ins)
        # convert to torch tensor
        img = self.transform_img(img)
        sem_seg = self.transform(img_t_cls)
        # aff_map = self.transform(aff_map)

        return img, sem_seg, aff_map

    def get_size(self,img_wh):
        width = img_wh[0]
        height = img_wh[1]
        if width < height:
            w = self.img_size
            h = int(self.img_size * height / width)
        else:
            h = self.img_size
            w = int(self.img_size * width / height)
        return w, h 

    def Affinity_generator(self, img_ins):
        """
        SSAP resolution 1/2, 1/4, 1/16, 1/32, 1/64 
        """
        # img_ins = Image.fromarray(img_ins)
        # 初始化一个aff_r * aff_r^2 * size * size
        aff_map = torch.zeros((self.aff_r, self.aff_r**2,self.img_size, self.img_size))
        ins_width, ins_height = img_ins.shape[0], img_ins.shape[1]

        for mul in range(self.aff_resolution):
            #resize大小后的ins, resize后的图片大小
            # ins_downsampe = cv2.resize(ins,cv2.INTER_NEAREST)
            img_t_aff_mul = img_ins[0:self.img_size:2**mul,0:self.img_size:2**mul]
            img_size = self.img_size // (2**mul)

            # 上下左右放大2个pixel
            img_t_aff_mul_2_pix = np.zeros((img_size + (self.aff_r//2)*2,
                                            img_size + (self.aff_r//2)*2, 3))
            img_t_aff_mul_2_pix[self.aff_r//2:img_size+self.aff_r//2,
                                self.aff_r//2:img_size+self.aff_r//2] \
                = img_t_aff_mul

            img_t_aff_compare = np.zeros((self.aff_r**2, img_size, img_size, 3))
            # 对25个affinity map进行错位填充ins
            for i in range(self.aff_r):
                for j in range(self.aff_r):
                    img_t_aff_compare[i*self.aff_r+j] = img_t_aff_mul_2_pix[i:i+img_size, j:j+img_size]

            # 相同物体affinity=1 不同affinity=0
            aff_data = np.where((img_t_aff_compare[:, :, :, 0] == img_t_aff_mul[:, :, 0])
                                & (img_t_aff_compare[:, :, :, 1] == img_t_aff_mul[:, :, 1])
                                & (img_t_aff_compare[:, :, :, 2] == img_t_aff_mul[:, :, 2]), 1, 0)
            aff_data = self.transform(aff_data.transpose(1, 2, 0))
            aff_map[mul, :, 0:img_size, 0:img_size] = aff_data
        return aff_map
    
    def Affinity_generator_new(self, img_ins):
        """
        SSAP resolution 1/2, 1/4, 1/16, 1/32, 1/64 
        """
        # img_ins = Image.fromarray(img_ins)
        # 初始化一个aff_r * aff_r^2 * size * size
        aff_map = np.zeros((self.aff_r, self.aff_r**2,self.img_size, self.img_size))
        ins_width, ins_height = img_ins.shape[0], img_ins.shape[1]

        for mul in range(self.aff_resolution):
            #resize大小后的ins, resize后的图片大小,instance最近邻插值
            img_size = self.img_size // (2**mul)
            ins_downsampe = cv2.resize(img_ins, (img_size,img_size), cv2.INTER_NEAREST)
            # tree-ins_downsampe
            
            #按affinity kernel半径padding
            ins_pad = cv2.copyMakeBorder(ins_downsampe,int(self.aff_r),int(self.aff_r),int(self.aff_r),int(self.aff_r),cv2.BORDER_CONSTANT, value=(0,0,0))
            aff_compare = np.zeros((self.aff_r**2, img_size, img_size, 3))
            # 对25个affinity kernel上进行错位填充ins
            for i in range(self.aff_r):
                for j in range(self.aff_r):
                    aff_compare[i*self.aff_r+j] = ins_pad[i:i+img_size, j:j+img_size]

            # 相同物体affinity=1 不同affinity=0
            aff_data = np.where((aff_compare[:, :, :, 0] == ins_downsampe[:, :, 0])
                                & (aff_compare[:, :, :, 1] == ins_downsampe[:, :, 1])
                                & (aff_compare[:, :, :, 2] == ins_downsampe[:, :, 2]), 1, 0)

            # aff_data = self.transform(aff_data.transpose(1, 2, 0))
            aff_map[mul, :, 0:img_size, 0:img_size] = aff_data
        return aff_map

def make_data_loader(args, train_dataset):
    train_sampler = None
    is_train=True
    distributed = True
    batch_size = args.batch_size // args.world_size
    if is_train:
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=1,
                                    drop_last=True,
                                    shuffle=False,
                                    pin_memory=True,
                                    sampler=train_sampler)
    
    return train_loader, train_sampler
# if __name__ == "__main__":
    

