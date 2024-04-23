import HZGammaAna
import time
import tensorflow as tf
start=time.time()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

print('GPU', tf.test.is_gpu_available())

#tf.config.experimental.set_memory_growth = True

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess = tf.compat.v1.Session(config = config)
#tf.compat.v1.keras.backend.set_session(sess)


import argparse
parser = argparse.ArgumentParser(description="Plotter for HZGamma analysis")
parser.add_argument('--batchsize', type=int, dest='batchsize', default=512, help="batchsize")
parser.add_argument('--depth', type=int, dest='depth', default=2, help="depth")
parser.add_argument('--nafdim', type=int, dest='nafdim', default=128, help="nafdim")
parser.add_argument('--permute', type=int, dest='permute', default=0, help='permute?')
parser.add_argument('--LRrange3_0', type=float, dest='LRrange3_0', default=0.001, help="LRrange3_0")
parser.add_argument('--LRrange3_1', type=float, dest='LRrange3_1', default=0.000001, help="LRrange3_1")
parser.add_argument('--LRrange3_2', type=float, dest='LRrange3_2', default=10000, help="LRrange3_2")
args = parser.parse_args()


#LRrange3 = [0.001, 0.000001, 10000, 0]
LRrange3 = [args.LRrange3_0, args.LRrange3_1, args.LRrange3_2, 0]

nafdim=args.nafdim
depth=args.depth
batchsize=args.batchsize
seed=123
step=500

if args.permute == 1:
    permute = True
else:
    permute = False

print('Start running')

HZGammaAna.train_and_validate(steps=step, LRrange=LRrange3, minibatch=batchsize, beta1=0.9, beta2=0.999, \
    #savedir=f'hzg_nafnodropout{nafdim}_batchsize{batchsize}_depth{depth}_seed{seed}_9_3_LR3mmd{LRrange3[0]}to{LRrange3[1]}s{LRrange3[2]}_condnet2_inputdim2_permute{permute}_region5/', nafdim=nafdim, depth=depth, seed=seed, permute=permute, retrain=False, train=False)
    savedir='test/', nafdim=nafdim, depth=depth, seed=seed, permute=permute, retrain=False, train=True)

end=time.time()
print('Running time: %s Seconds'%(end-start))