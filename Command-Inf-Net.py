import os
import argparse

import MyTrain_LungInf as mTrain

os.system('python MyCVk10.py --epoch 2 --batchsize 4 --num_workers 2 --k_size 2 --train_path ./Dataset/CV')
#mTrain.main(['--batchsize', '6', '--num_workers', '4', '--backbone', 'Res2Net50', '--train_path', './Dataset/TrainingSet/LungSegmentation/Montgomery'])




