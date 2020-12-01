import numpy as np
import torch
import os
import cv2
from models.GT import GT_Model
from models.GR import GR_Model
from models.encoder_build import encoder
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from utils import index
import argparse
from collections import OrderedDict as odict


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def str2bool(v):
    return v.lower() in ('y', 'yes', 't', 'true', '1')

parser = argparse.ArgumentParser('test')
parser.add_argument('--save_result_path',type=str2bool,default=True,help="if save result")
parser.add_argument('--real20',type=str2bool,default=True,help="if real20 test")
parser.add_argument('--sir_wild',type=str2bool,default=False,help="if sir_wild test")
parser.add_argument('--sir_solid',type=str2bool,default=False,help="if sir_solid test")
parser.add_argument('--sir_postcard',type=str2bool,default=False,help="if sir_postcard test")
parser.add_argument('--real45',type=str2bool,default=True,help="if real45 test")
parser.add_argument('--num_workers',type=int,default=4,help="num_workers")
args = parser.parse_args()
    
#dataset without GT
path_real45 = './testsets/real45/'

#datasets with GT
path_real20 = './testsets/real20/' 
# The following datasets are not provided due to their policy
# You can apply for the SIR^2 dataset from https://sir2data.github.io/
path_sir_wild = './testsets/WildSceneDataset/withgt/'
path_sir_postcard = './testsets/PostcardDataset/' 
path_sir_solid = './testsets/SolidObjectDataset/'


encoder_I = encoder()
encoder_R = encoder()
encoder_I.cuda()
encoder_R.cuda()
gt_model = GT_Model(encoder_I,encoder_R)
gt_model.cuda()
gt_model.eval()

encoder = encoder()
encoder.cuda()
gr_model = GR_Model(encoder)
gr_model.cuda()
gr_model.eval()

def creat_list(path,if_gt=True):
    gt_list = []
    image_list = []
                   
    if if_gt:
        blended_path = path + 'blended/'
        trans_path = path + 'transmission_layer/'
        for _,_,fnames in sorted(os.walk(blended_path)):
            for fname in fnames:
                image_list.append(blended_path+fname)
                gt_list.append(trans_path+fname)
                
    else:
        for _,_,fnames in sorted(os.walk(path)):
            for fname in fnames:
                image_list.append(path+fname)
                
    return image_list,gt_list

class TestDataset(Dataset):
    def __init__(self,blended_list,trans_list,transform=False,if_GT=True):
        self.to_tensor = transforms.ToTensor()            
        self.blended_list = blended_list
        self.trans_list = trans_list
        self.transform = transform
        self.if_GT = if_GT

    def __getitem__(self, index):
        blended = cv2.imread(self.blended_list[index])
        trans = blended
        if self.if_GT:
            trans= cv2.imread(self.trans_list[index])
        if self.transform == True:
            if trans.shape[0] > trans.shape[1]:
                neww = 300
                newh = round((neww / trans.shape[1]) * trans.shape[0])
            if trans.shape[0] < trans.shape[1]:
                newh = 300
                neww = round((newh / trans.shape[0]) * trans.shape[1])
            blended = cv2.resize(np.float32(blended), (neww, newh), cv2.INTER_CUBIC)/255.0
            trans = cv2.resize(np.float32(trans), (neww, newh), cv2.INTER_CUBIC)/255.0

        blended = self.to_tensor(blended)
        trans = self.to_tensor(trans)
        return blended,trans
    
    def __len__(self):
        return len(self.blended_list)

def test_diffdataset(test_loader,save_path=None,if_GT=True):
    ssim_sum = 0
    psnr_sum = 0
    
    for j, (image, gt) in enumerate(test_loader):
        image = image.cuda()
        gt = gt.cuda()
        with torch.no_grad():   
            image.requires_grad_(False)
            
            pretrained_R = gr_model(image)
            output_t, *_ = gt_model(image, pretrained_R)

            output_t = index.tensor2im(output_t)
            gt = index.tensor2im(gt)
            pretrained_R = index.tensor2im(pretrained_R)
            
            if if_GT:
                res, psnr,ssim = index.quality_assess(output_t, gt)
                print(res)
                ssim_sum += ssim
                psnr_sum += psnr 
                
            if save_path:
                if not os.path.exists(save_path):
                    os.mkdir(save_path) 
                image = index.tensor2im(image)
                cv2.imwrite("%s/%s_pretr.png"%(save_path,j),pretrained_R,[int(cv2.IMWRITE_JPEG_QUALITY),100])   
                cv2.imwrite("%s/%s_b.png"%(save_path,j),image,[int(cv2.IMWRITE_JPEG_QUALITY),100])  
                cv2.imwrite("%s/%s_t.png"%(save_path,j),output_t,[int(cv2.IMWRITE_JPEG_QUALITY),100])     
                if if_GT:
                    cv2.imwrite("%s/%s_gt.png"%(save_path,j),gt,[int(cv2.IMWRITE_JPEG_QUALITY),100])     
                
    print(len(test_loader),'SSIM:',ssim_sum/len(test_loader),'PSNR:',psnr_sum/len(test_loader))
    return len(test_loader),ssim_sum,psnr_sum


def test_state(state_T,state_R):
  
    gt_model.load_state_dict(state_T)
    gr_model.load_state_dict(state_R)
    del(state_T)
    del(state_R)

    datasets = odict([('real20', True), ('real45', False), ('sir_wild', True), ('sir_solid', True), ('sir_postcard', True)])

    psnr_all, ssim_all, num_all = 0, 0, 0
    for dataset, with_GT in datasets.items():
        if getattr(args, dataset):
            if args.save_result_path:
                save_path = './result/' + dataset
            else:
                save_path = None
            print('testing dataset:',dataset)
            num, ssim_sum, psnr_sum = test_diffdataset(eval('test_loader_'+dataset.replace('_','')), save_path, with_GT)
            
            if with_GT:
                psnr_all += psnr_sum
                ssim_all += ssim_sum
                num_all += num
    ssim_av = ssim_all/num_all
    psnr_av = psnr_all/num_all
    return psnr_av,ssim_av


image_list_real20, gt_list_real20 = creat_list(path_real20)
test_dataset_real20 = TestDataset(image_list_real20, gt_list_real20,transform=True)
test_loader_real20 = torch.utils.data.DataLoader(dataset=test_dataset_real20,\
                                                 batch_size=1,shuffle=False,num_workers=args.num_workers)

image_list_sirwild, gt_list_sirwild = creat_list(path_sir_wild)
test_dataset_sirwild = TestDataset(image_list_sirwild,gt_list_sirwild)
test_loader_sirwild = torch.utils.data.DataLoader(dataset=test_dataset_sirwild,\
                                                  batch_size=1,shuffle=False,num_workers=args.num_workers)

image_list_sirpostcard, gt_list_sirpostcard = creat_list(path_sir_postcard)
test_dataset_sirpostcard = TestDataset(image_list_sirpostcard,gt_list_sirpostcard)
test_loader_sirpostcard = torch.utils.data.DataLoader(dataset=test_dataset_sirpostcard,\
                                                      batch_size=1,shuffle=False,num_workers=args.num_workers)

image_list_sirsolid, gt_list_sirsolid = creat_list(path_sir_solid)
test_dataset_sirsolid = TestDataset(image_list_sirsolid,gt_list_sirsolid)
test_loader_sirsolid = torch.utils.data.DataLoader(dataset=test_dataset_sirsolid,\
                                                   batch_size=1,shuffle=False,num_workers=args.num_workers)

image_list_real45, gt_list_real45 = creat_list(path_real45,if_gt=False)
test_dataset_real45 = TestDataset(image_list_real45,gt_list_real45,if_GT=False)
test_loader_real45 = torch.utils.data.DataLoader(dataset=test_dataset_real45,\
                                                 batch_size=1,shuffle=False,num_workers=args.num_workers)

if __name__ == '__main__':
    
    ckpt_path  = './checkpoint/pretrain.pth'
    ckpt_pre = torch.load(ckpt_path)
    print("loading checkpoint'{}'".format(ckpt_path))
    
    psnr_av,ssim_av = test_state(ckpt_pre['GT_state'],ckpt_pre['GR_state'])    
    print('The average PSNR/SSIM of all chosen testsets:',psnr_av,ssim_av) 
    

