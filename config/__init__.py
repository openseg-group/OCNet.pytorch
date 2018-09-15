import argparse
import os
import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Parameters():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch Segmentation Network")
        parser.add_argument("--dataset", type=str, default="cityscapes_train",
                            help="Specify the dataset to use.")
        parser.add_argument("--batch-size", type=int, default=8,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--data-dir", type=str, default='/teamscratch/msravcshare/yuyua/deeplab_v3/dataset/cityscapes',
                            help="Path to the directory containing the PASCAL VOC dataset.")
        parser.add_argument("--data-list", type=str, default='./dataset/list/cityscapes/train.lst',
                            help="Path to the file listing the images in the dataset.")
        parser.add_argument("--ignore-label", type=int, default=255,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--input-size", type=str, default='769,769',
                            help="Comma-separated string with height and width of images.")
        parser.add_argument("--is-training", action="store_true",
                            help="Whether to updates the running means and variances during the training.")
        parser.add_argument("--learning-rate", type=float, default=1e-2,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=0.9,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--not-restore-last", action="store_true",
                            help="Whether to not restore last (FC) layers.")
        parser.add_argument("--num-classes", type=int, default=19,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--start-iters", type=int, default=0,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--num-steps", type=int, default=40000,
                            help="Number of training steps.")
        parser.add_argument("--power", type=float, default=0.9,
                            help="Decay parameter to compute the learning rate.")
        parser.add_argument("--random-mirror", action="store_true",
                            help="Whether to randomly mirror the inputs during the training.")
        parser.add_argument("--random-scale", action="store_true",
                            help="Whether to randomly scale the inputs during the training.")
        parser.add_argument("--random-seed", type=int, default=304,
                            help="Random seed to have reproducible results.")
        parser.add_argument("--restore-from", type=str, default='./pretrain_model/MS_DeepLab_resnet_pretrained_COCO_init.pth',
                            help="Where restore model parameters from.")
        parser.add_argument("--save-num-images", type=int, default=2,
                            help="How many images to save.")
        parser.add_argument("--save-pred-every", type=int, default=5000,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--snapshot-dir", type=str, default='./snapshots_psp_ohem_trainval/',
                            help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=5e-4,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--gpu", type=str, default='0',
                            help="choose gpu device.")

        parser.add_argument("--ohem-thres", type=float, default=0.6,
                            help="choose the samples with correct probability underthe threshold.")
        parser.add_argument("--ohem-thres1", type=float, default=0.8,
                            help="choose the threshold for easy samples.")
        parser.add_argument("--ohem-thres2", type=float, default=0.5,
                            help="choose the threshold for hard samples.")
        parser.add_argument("--use-weight", type=str2bool, nargs='?', const=True,
                            help="whether use the weights to solve the unbalance problem between classes.")
        parser.add_argument("--use-val", type=str2bool, nargs='?', const=True,
                            help="choose whether to use the validation set to train.")
        parser.add_argument("--use-extra", type=str2bool, nargs='?', const=True,
                            help="choose whether to use the extra set to train.")
        parser.add_argument("--ohem", type=str2bool, nargs='?', const=True,
                            help="choose whether conduct ohem.")
        parser.add_argument("--ohem-keep", type=int, default=100000,
                            help="choose the samples with correct probability underthe threshold.")
        parser.add_argument("--network", type=str, default='resnet101',
                            help="choose which network to use.")
        parser.add_argument("--method", type=str, default='base', 
                            help="choose method to train.")
        parser.add_argument("--reduce", action="store_false",
                            help="Whether to use reduce when computing the cross entropy loss.")
        parser.add_argument("--ohem-single", action="store_true",
                            help="Whether to use hard sample mining only for the last supervision.")
        parser.add_argument("--use-parallel", action="store_true",
                            help="Whether to the default parallel.")
        parser.add_argument("--dsn-weight", type=float, default=0.4,
                            help="choose the weight of the dsn supervision.")
        parser.add_argument("--pair-weight", type=float, default=1,
                            help="choose the weight of the pair-wise loss supervision.")
        parser.add_argument('--seed', default=304, type=int, help='manual seed')

        parser.add_argument("--output-path", type=str, default='./seg_output_eval_set',
                        help="Path to the segmentation map prediction.")
        parser.add_argument("--store-output", type=str, default='False',
                        help="whether store the predicted segmentation map.")
        parser.add_argument("--use-flip", type=str, default='False',
                        help="whether use test-stage flip.")
        parser.add_argument("--use-ms", type=str, default='False',
                        help="whether use test-stage multi-scale crop.")
        parser.add_argument("--predict-choice", type=str, default='whole',
                        help="crop: choose the training crop size; whole: choose the whole picture; step: choose to predict the images with multiple steps.")
        parser.add_argument("--whole-scale", type=str, default='1',
                        help="choose the scale to rescale whole picture.")

        parser.add_argument("--start-epochs", type=int, default=0,
                            help="Number of the initial staring epochs.")
        parser.add_argument("--end-epochs", type=int, default=120,
                            help="Number of the overall training epochs.")
        parser.add_argument("--save-epoch", type=int, default=20,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--criterion", type=str, default='ce',
                        help="Specify the specific criterion/loss functions to use.")
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument("--fix-lr", action="store_true",
                            help="choose whether to fix the learning rate.")
        parser.add_argument('--log-file', type=str, default= "", 
                            help='the output file to redirect the ouput.')

        parser.add_argument("--use-normalize-transform", action="store_true",
                            help="Whether to the transform the input data by mean, variance.")        
        self.parser = parser


    def parse(self):
        args = self.parser.parse_args()
        return args
