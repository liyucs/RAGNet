import os
import argparse
from solver import Solver

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser('Perceptual Reflection Removel')

parser.add_argument('--ref_syn_dir',default="./synthetic",help="path to synthetic data")
parser.add_argument('--ref_real_dir',default="./real",help="path to real data")
parser.add_argument('--vgg_feature_pretrained_ckpt',default='./checkpoint/vgg19-dcbb9e9d.pth',help='path to vgg_19 ckpt')
parser.add_argument('--save_model_freq',default=5,type=int,help="frequency to save model")
parser.add_argument('--print_freq',type=int,default=10,help='print frequency')
parser.add_argument('--resume_file',default='',help="resume file path")
parser.add_argument('--lr',default=1e-4,type=float,help="learning rate")
parser.add_argument('--load_workers',default=4,type=int,help="number of workers to load data")      
parser.add_argument('--batch_size',default=1,type=int,help="batch size")
parser.add_argument('--start_epoch',type=int,default=0,help="start epoch of training")
parser.add_argument('--num_epochs',type=int,default=150,help="total epoch of training")

def main():
	args = parser.parse_args()
	solver=Solver(args) 
	solver.train_model()

if __name__=='__main__':
	main()
