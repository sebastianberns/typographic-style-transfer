#!/usr/bin/env python

import atexit
from argparse import ArgumentParser
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import BinaryAccuracy, Loss

import config
from generic import Invert, MCGAN_dataset, Supermodel, Logger, Plotter
import style.models
from style.data import TrainLoader, EvalLoader


def get_data_loaders(train_datadir, train_batchsize, val_datadir, val_batchsize, use_cuda):
    # DATA PREPROCESSING
    train_feature_transforms = tv.transforms.Compose([
        Invert(),
        tv.transforms.ToTensor()
    ])
    val_feature_transforms = tv.transforms.Compose([
        tv.transforms.Grayscale(),
        Invert(),
        tv.transforms.ToTensor()
    ])
    # Resolve classes: get the class directory name and convert into float
    # (this way we can store positve samples in '1/' and negative in '0/')
    val_target_transforms = tv.transforms.Lambda(lambda c: np.array([val_dataset.classes[c]], dtype=np.float32))
    
    # DATASETS
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_dataset = MCGAN_dataset(train_datadir, config, transform=train_feature_transforms)
    train_loader = TrainLoader(train_dataset, config, batch_size=train_batchsize,
                               shuffle=True, **kwargs)

    val_dataset = tv.datasets.ImageFolder(val_datadir, transform=val_feature_transforms,
                                          target_transform=val_target_transforms)
    val_loader = EvalLoader(val_dataset, config, batch_size=val_batchsize,
                            shuffle=True, **kwargs)
    
    return train_loader, val_loader


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--epochs', type=int, default=10,
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--model', type=str, default=None,
        help='network model name')
    parser.set_defaults(autosave=True)
    parser.add_argument(
        '--disable_autosave', dest='autosave', action='store_false',
        help='disable saving of model states')
    parser.set_defaults(autodelete=True)
    parser.add_argument(
        '--disable_autodelete', dest='autodelete', action='store_false',
        help='disable deletion of previous model states')

    parser.add_argument(
        '--train_datadir', type=str, default='./train',
        help='training data directory (default: ./train)')
    parser.add_argument(
        '--train_batchsize', type=int, default=128,
        help='input batch size for training (default: 128)')

    parser.add_argument(
        '--val_datadir', type=str, default='./val',
        help='validation data directory (default: ./val)')
    parser.add_argument(
        '--val_batchsize', type=int, default=128,
        help='input batch size for validation (default: 128)')

    parser.set_defaults(cuda=True)
    parser.add_argument(
        '--disable_cuda', dest='cuda', action='store_false',
        help='Disable use of CUDA')
    parser.add_argument(
        '--seed', type=int, default=int(time.time()),
        help='random seed (default: time.time())')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    starttime = time.time()
    
    # Set up logging and plotting
    logfile = 'log.csv'
    logheader = ("Epoch", "Train Loss", "Train Error", "Val Loss", "Val Error")
    plots = [
        ("plot_error.pdf", {'x':logheader[0], 'y':[logheader[2], logheader[4]], 'ylim':(0,1)}),
        ("plot_loss.pdf",  {'x':logheader[0], 'y':[logheader[1], logheader[3]]})
    ]
    logger = Logger(*logheader, file=logfile)
    plotter = Plotter(logfile, plots)
    atexit.register(plotter.save) # plot on exit
    
    args = get_args()
    set_random_seed(args.seed)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    train_loader, val_loader = get_data_loaders(args.train_datadir, args.train_batchsize,
                                                args.val_datadir, args.val_batchsize,
                                                args.cuda)
    
    classifier = Supermodel(style.models.index, args.model).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(classifier.model.parameters(), lr=args.lr, momentum=0.95)
    
    trainer = create_supervised_trainer(classifier, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(classifier, metrics={
                                                'accuracy': BinaryAccuracy(),
                                                'loss': Loss(criterion)
                                            }, device=device)
    
    @trainer.on(Events.STARTED)
    def init(engine):
        print("Timestamp {}".format(starttime), "\n")
        for arg in vars(args): # Arguments
            print(arg, getattr(args, arg))
        print()
        print(classifier, "\n") # Model
        print("Startup took {:.3f} sec".format(time.time()-starttime))
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        # Evaluation on training set
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        train_accuracy = train_metrics['accuracy']
        train_error = 1. - train_accuracy
        train_loss = train_metrics['loss']
        
        # Evaluation on validation set
        evaluator.run(val_loader)
        val_metrics = evaluator.state.metrics
        val_accuracy = val_metrics['accuracy']
        val_error = 1. - val_accuracy
        val_loss = val_metrics['loss']
        
        logger.update(engine.state.epoch, train_loss, train_error, val_loss, val_error)
        
        print("Epoch {:>4}\t"
              "Train Loss  {:.4f}\t"
              "Train Acc  {:.2f}%\t"
              "Val Loss  {:.4f}\t"
              "Val Acc  {:.2f}%\t"
              "\tTime  {:.0f} sec\t".format(
                engine.state.epoch,
                train_loss,
                train_accuracy*100,
                val_loss,
                val_accuracy*100,
                time.time()-starttime))
        
        if val_error < classifier.best[1] and args.autosave:
            savetimer = time.time()
            classifier.best = (engine.state.epoch, val_error, val_loss)
            saved = classifier.save(engine.state.epoch, rmprev=args.autodelete)
            print("New best (val error {}%), model state saved as '{}' ({:.3f} sec)".format(
                val_error*100,
                saved,
                time.time()-savetimer))
    
    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
