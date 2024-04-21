import pprint
from train import Train
from image_preprocessor import ImagePreprocessor
from cv2 import face
import cv2 as cv
import numpy as np
import sklearn.metrics


'''
def eigen(args):
    model = face.EigenFaceRecognizer.create()
    if args.model != None:
        model = face.EigenFaceRecognizer.read(str(args.model))
    common(args, model)
'''


def eigen(args):
    model = face.EigenFaceRecognizer.create()
    if args.model is not None:
        model.read(str(args.model))
    else:
        common(args, model)

    eigenvalues = model.getEigenValues()
    eigenvectors = model.getEigenVectors()
    print("Eigenvalues:")
    print(eigenvalues)
    np.savetxt("eigenvalues.csv", eigenvalues, delimiter=",")

    mean = model.getMean()
    np.savetxt("meanface.csv", mean, delimiter=",")

    for i, eig_face in enumerate(eigenvectors):
        np.savetxt(f"eigenface{i + 1}.csv", eig_face, delimiter=",")

def fisher(args):
    model = face.FisherFaceRecognizer.create()
    if args.model is not None:
        model.read(str(args.model))
    common(args, model)

def lbph(args):
    model = face.LBPHFaceRecognizer.create()
    if args.model is not None:
        model.read(str(args.model))
    common(args, model)

def common(args, model):
    train = Train(args.data, ImagePreprocessor(), args.split)
    metric_options = {"accuracy": sklearn.metrics.accuracy_score}
    metrics = [metric_options[m] for m in args.metric]
    if args.cross is not None:
        for name, metric in zip(args.metric, metrics):
            res = train.cross_validate(model, args.cross, metric)
            print('{metric}:'.format(metric=name))
            pprint.pprint(res)
            print('\n')
    if args.model is None:
        train.train(model)
    if args.save:
        model.write('model.yaml')

import argparse
import pathlib

parentParser = argparse.ArgumentParser(add_help=False)
parentParser.add_argument('data', type=pathlib.Path, nargs='+', help='path to data')
parentParser.add_argument('-s', '--split', type=float, help='train-test split', default=0.5)
parentParser.add_argument('-c', '--cross', type=int, help='cross validation')
parentParser.add_argument('-ns', '--no-save', dest='save', help='do not save model', action='store_false', default=True)
parentParser.add_argument('-m', '--metric', help='metric(s) to evaluate', nargs='+', choices=['accuracy'], default=['accuracy'])
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
