import pprint
from train import Train
from image_preprocessor import ImagePreprocessor
import cv2 as cv
import sklearn.metrics
import numpy as np
import os

def eigen(args):
    model = cv.face.EigenFaceRecognizer.create()
    if args.components != None:
        model = cv.face.EigenFaceRecognizer.create(args.components)
    if args.model != None:
        model = cv.face.EigenFaceRecognizer.read(str(args.model))
    train = common(args, model)
    eigenvalues = model.getEigenValues()
    eigenvectors = model.getEigenVectors().T
    mean = model.getMean()
    if 'mean' in args.esave:
        np.save('mean.npy', mean)
    if 'eva' in args.esave:
        np.savetxt('eigenvalues.txt', eigenvalues, delimiter=',')
    if 'eve' in args.esave:
        os.mkdir('eigenfaces')
        for i in range(model.getNumComponents()):
            ev = eigenvectors[i, :]
            ev = ev.reshape(train.dimensions)
            normalized = np.zeros(ev.shape)
            cv.normalize(ev, normalized, 0, 255, cv.NORM_MINMAX)
            colored = cv.applyColorMap(normalized.astype(np.uint8), cv.COLORMAP_JET)
            cv.imwrite('eigenfaces/eigenface{i}.png'.format(i=i), colored)


def fisher(args):
    model = cv.face.FisherFaceRecognizer.create()
    if args.model != None:
        model = cv.face.FisherFaceRecognizer.read(str(args.model))
    common(args, model)

def lbph(args):
    model = cv.face.LBPHFaceRecognizer.create()
    if args.model != None:
        model = cv.face.LBPHFaceRecognizer.read(str(args.model))
    common(args, model)

def common(args, model):
    train = Train(args.data, ImagePreprocessor(), args.split)
    metric_options = {"accuracy": sklearn.metrics.accuracy_score}
    metrics = [metric_options[m] for m in args.metric]
    if args.cross != None:
        for name, metric in zip(args.metric, metrics):
            res = train.cross_validate(model, args.cross, metric)
            print('{metric}:'.format(metric=name))
            pprint.pprint(res)
            print('\n')
    if args.model == None:
        train.train(model)
    if (args.save):
        model.write('model.yaml')
    return train
    

import argparse
import pathlib

parentParser = argparse.ArgumentParser(add_help=False)
parentParser.add_argument('data', type=pathlib.Path, nargs='+', help='path to data')
parentParser.add_argument('-s', '--split', type=float, help='train-test split', default=0)
parentParser.add_argument('-c', '--cross', type=int, help='cross validation')
parentParser.add_argument('-ns', '--no-save', dest='save', help='do not save model', action='store_false',
                          default=True)
parentParser.add_argument('-m', '--metric', help='metric(s) to evaluate', nargs='+', choices=['accuracy'],
                          default=['accuracy'])
parentParser.add_argument('-mo', '--model', type=pathlib.Path, help='path to saved model')

parser = argparse.ArgumentParser(description='Facial recognition with OpenCV')
subparsers = parser.add_subparsers(help='subcommands', required=True)

parser_eigen = subparsers.add_parser('eigen', parents=[parentParser], description='Eigenfaces')
parser_eigen.set_defaults(func=eigen)
parser_eigen.add_argument('-p', '--components', type=int, help='number of principal components')
parser_eigen.add_argument('-e', '--save', dest='esave', nargs='+', choices=['mean', 'eva', 'eve'])

parser_fisher = subparsers.add_parser('fisher', parents=[parentParser], description='Fisherfaces')
parser_fisher.set_defaults(func=fisher)

parser_lbph = subparsers.add_parser('lbph', parents=[parentParser], description='LBPH')
parser_lbph.set_defaults(func=lbph)
    
args = parser.parse_args('eigen C:/Users/nickl/COP4930/final_project/small_images -s .2 -p 10 -e eva eve mean'.split())
args.func(args)