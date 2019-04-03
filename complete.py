#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import cv2
import tensorflow as tf
import siftAccu as sift 
from model import DCGAN
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
parser = argparse.ArgumentParser()
parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--outInterval', type=int, default=50)
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres','specific'],
                    default='specific')
parser.add_argument('--centerScale', type=float, default=0.25)
parser.add_argument('--imgs', type=str, nargs='+',default="C:/Users\\engab\\Desktop\\PROJECT\\tensorflow dcgan inpainting\\data\\imagesToComplete")

args = parser.parse_args()
assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  batch_size=min(64, len(args.imgs)),
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    img1 = cv2.imread("C:/Users/engab/Desktop/PROJECT/CRW_4901_JFRtamp37.jpg",cv2.IMREAD_COLOR)
    points,positions=sift.all_experiments(img1)   

    while True:
     
     dcgan.complete(args,points,positions,110,175)#put the starting coordinates of the object
     output=cv2.imread("C:/Users/engab/Desktop/PROJECT/tensorflow dcgan inpainting/DSC_1535tamp1.jpg",cv2.IMREAD_COLOR)#inpainted image path
     img1[175:239,110:174]=output
     points,positions=sift.all_experiments(img1)
     #your code here,put image number 950 in the loop folder
     parser.set_defaults(imgs="C:/Users\\engab\\Desktop\\PROJECT\\tensorflow dcgan inpainting\\data\\loop")#inpainted image path
     args = parser.parse_args()
     if points<4:
      break
     
    
