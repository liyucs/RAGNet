import cv2
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from utils.utils import syn
import torchvision.transforms as transforms  

def Crop_img(img):    
    crop_w = img.shape[0] - 224
    crop_h = img.shape[1] - 224
    if crop_w == 0:
        random_w = 0
    else:
        random_w = int(np.random.randint(0,crop_w)/2)
    if crop_h == 0:
        random_h = 0
    else:
        random_h = int(np.random.randint(0,crop_h)/2)
            
    return img[random_w:random_w+224,random_h:random_h+224]

to_tensor = transforms.ToTensor()    


class MyDataset(Dataset):
    def __init__(self,dir_b_list,dir_t_list,dir_r_list,is_ref_syn,crop=Crop_img):
       
        self.dir_b_list=dir_b_list
        self.dir_t_list=dir_t_list
        self.dir_r_list=dir_r_list
        self.is_ref_syn=is_ref_syn
        self.crop = Crop_img
        

    def __getitem__(self, index):
        
        t_img = self.dir_t_list[index]
            
        if self.is_ref_syn:
            r_img=self.dir_r_list[index]
            
            oh = t_img.shape[0]
            ow = t_img.shape[1]
            new = int(np.random.randint(224,448)/2)*2
            neww = round((new / t_img.shape[0]) * t_img.shape[1])
            newh = round((new / t_img.shape[1]) * t_img.shape[0])
            if ow >= oh:
                r_img = cv2.resize(np.float32(r_img), (neww, new), cv2.INTER_CUBIC)
                t_img = cv2.resize(np.float32(t_img), (neww, new), cv2.INTER_CUBIC)
            if oh > ow:
                r_img = cv2.resize(np.float32(r_img), (new, newh), cv2.INTER_CUBIC)
                t_img = cv2.resize(np.float32(t_img), (new, newh), cv2.INTER_CUBIC)
            
            r_img = self.crop(r_img)
            t_img = self.crop(t_img)

            t_img,r_img,b_img= syn(t_img,r_img)

            
        else:
            b_img = self.dir_b_list[index]
            oh = t_img.shape[0]
            ow = t_img.shape[1]
            new = int(np.random.randint(224,480)/2)*2
            neww = round((new / t_img.shape[0]) * t_img.shape[1])
            newh = round((new / t_img.shape[1]) * t_img.shape[0])
            if ow >= oh:
                t_img_ = cv2.resize(np.float32(t_img), (neww, new), cv2.INTER_CUBIC)/255.0
                b_img_ = cv2.resize(np.float32(b_img), (neww, new), cv2.INTER_CUBIC)/255.0
                if new == 224:
                    randh = 0
                else:
                    randh = int(np.random.randint(0,new-224))
                if neww == 224:
                    randw = 0
                else:
                    randw = int(np.random.randint(0,neww-224))
                t_img = t_img_[randh:randh+224,randw:randw+224]
                b_img = b_img_[randh:randh+224,randw:randw+224]
            if oh > ow:
                t_img_ = cv2.resize(np.float32(t_img), (new, newh), cv2.INTER_CUBIC)/255.0
                b_img_ = cv2.resize(np.float32(b_img), (new, newh), cv2.INTER_CUBIC)/255.0
                if new == 224:
                    randw = 0
                else:
                    randw = int(np.random.randint(0,new-224))
                if newh == 224:
                    randh = 0
                else:
                    randh = int(np.random.randint(0,newh-224))
                t_img = t_img_[randh:randh+224,randw:randw+224]
                b_img = b_img_[randh:randh+224,randw:randw+224]
            r_img=t_img.copy()
            
       
        b_img = to_tensor(b_img)
        r_img = to_tensor(r_img)
        t_img = to_tensor(t_img)

        return b_img,t_img,r_img,self.is_ref_syn

    def __len__(self):
        return len(self.dir_t_list)
        
class FusionDataset(Dataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1. / len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' % (
        self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size
        
    
    
    
    
    
    
    
    
       
        