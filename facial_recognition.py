from train import Train
from image_preprocessor import ImagePreprocessor
from cv2 import face

def eigen(args):
    model = face.EigenFaceRecognizer.create()
    common(args, model)


def fisher(args):
    model = face.FisherFaceRecognizer.create()
    common(args, model)

def lbph(args):
    model = face.LBPHFaceRecognizer.create()
    common(args, model)

def common(args, model):
    train = Train(args.data, model, ImagePreprocessor())
    if (args.cross):
        train.cross_validate()

import argparse
import pathlib

parentParser = argparse.ArgumentParser(add_help=False)
parentParser.add_argument('data', type=pathlib.Path, nargs='+', help='path to data')
parentParser.add_argument('-s', '--split', type=float help='train-test split', default=0)
parentParser.add_argument('-c', '--cross', type=int, help='cross validation', default=0)
parentParser.add_argument('-ns', '--no-save', dest='save', help='do not save model', action='store_false',
                          default=True)
parentParser.add_argument('-m', '--metric', help='metric(s) to evaluate', nargs='+', choices=['accuracy'])
parentParser.add_argument('-mo', '--model', type=pathlib.Path, help='path to saved model')

parser = argparse.ArgumentParser(description='Facial recognition with OpenCV')
subparsers = parser.add_subparsers(help='subcommands', required=True)

parser_eigen = subparsers.add_parser('eigen', parents=[parentParser], description='Eigenfaces')
parser_eigen.set_defaults(func=eigen)

parser_fisher = subparsers.add_parser('fisher', parents=[parentParser], description='Fisherfaces')
parser_fisher.set_defaults(func=fisher)

parser_lbph = subparsers.add_parser('lbph', parents=[parentParser], description='LBPH')
parser_lbph.set_defaults(func=lbph)

args = parser.parse_args()
args.func(args)