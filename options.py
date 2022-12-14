import argparse
import const


def options():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--dataset', dest='dataset', choices=const.DATASETS,
                        default='CIFAR10', type=str)
    parser.add_argument('--model', dest='model', choices=const.MODELS,
                        default='resnet18', type=str,
                        help='datasets')
    parser.add_argument('--bs', dest='batch_size', choices=const.BATCHSIZE,
                        default=32, type=int,
                        help='batch_size')
    parser.add_argument('--epochs', dest='epochs',
                        default=30, type=int,
                        help='epochs of training')
    parser.add_argument('--lr', dest='lr',
                        default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--streak', dest='streak',
                        default=5, type=int,
                        help='memorization streak number for an example to be removed')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=5, type=int,
                        help='epoch interval to evaluate the memorized data')
    parser.add_argument('--es', dest='early_stop',
                        default=False, action="store_true",
                        help='early stop or not')
    parser.add_argument('--baseline', dest='baseline',
                        default=False, action="store_true",
                        help='train without any modification')
    parser.add_argument('--return_memorized_fraction', dest='return_memorized_fraction',
                        default=False, action="store_true",
                        help='return the fraction of the truly memorized samples')
    parser.add_argument('--remove_indices', dest='remove_indices',
                        default=None, type=str,
                        help='path to file with indices to remove')
    parser.add_argument('--remove_fraction', dest='remove_fraction',
                        default=False, action="store_true",
                        help='remove only a fraction of the memrozed indices while retraining')
    parser.add_argument('--nt',dest='noise_type',
                    default='symmetric',type=str,
                    help='noise type: symmetric/asymmetric')
    parser.add_argument('--k',dest='top_k',
                    default=0.05,type=float,
                    help='noise fraction 0-1')
    parser.add_argument('--ls',dest='label_shuffle',
                    default=True,type=bool,
                    help='wheter to shuffle labels of k percent of training samples')
    args = parser.parse_args()
    return args
