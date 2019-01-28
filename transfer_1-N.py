#!/usr/bin/env python

import atexit
from argparse import ArgumentParser
import random
import shutil
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

import config
from generic import Invert, Supermodel, Logger, Plotter
import style.models
from transfer.data import load_font_abc, batch_aabbcc, load_font_abcabc, rand_glyphs, savedata
from transfer.loss import greymean_loss, greyratio_loss, tvloss_abs, tvloss_sq


batchsize = config.LETTERS**2 # 676
savedir = "optim"

available_losses = [
    'LOSS_SIM',
    'LOSS_DISSIM',
    'LOSS_GRMEAN',
    'LOSS_GRRATIO',
    'LOSS_TVABS',
    'LOSS_TVSQ'
]


def get_args():
    parser = ArgumentParser()

    # Network
    parser.add_argument(
        '--model', type=str, default=None,
        help='Network model name')
    parser.add_argument(
        '--state', type=str, default=None,
        help='Path to network state dict')
    
    parser.set_defaults(cuda=True)
    parser.add_argument(
        '--disable_cuda', dest='cuda', action='store_false',
        help='Disable use of CUDA')
    parser.add_argument(
        '--seed', type=int, default=int(time.time()),
        help='Random seed (default: time.time())')
    
    # Optimizer
    parser.add_argument(
        '--steps', type=int, default=10,
        help='Number of optimization steps (default: 10)')
    parser.add_argument(
        '--lr', type=float, default=1.0,
        help='Learning rate (default: 1.0)')
    
    parser.add_argument(
        '--greymean', type=float, default=0.5,
        help='Mean grey pixel value (default: 0.5)')
    parser.add_argument(
        '--greyratio', type=float, default=0.14,
        help='Mean grey pixel ratio (default: 0.14)')
    
    # Transfer
    parser.add_argument(
        '--contentfont', type=str, default=None,
        help='Path to content font')
    parser.add_argument(
        '--stylefont', type=str, default=None,
        help='Path to style font')
    
    # Loss
    for loss in available_losses:
        parser.add_argument(
            '--{}'.format(loss), type=float, nargs='?', const=1, default=0,
            help='{} default weight: 0'.format(loss))
    
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
    print('Timestamp {}'.format(starttime), '\n')
    
    args = get_args()
    for arg in vars(args): # Arguments
        print(arg, getattr(args, arg))
    print()
    
    # Set up logging and plotting
    logfile = 'log.csv'
    logheader = ("Step",
                 "Avg Style Font Classification", "Avg Content Font Classification",
                 "Avg Similarity Loss", "Avg Dissimilarity Loss",
                 "Avg Grey Mean Loss", "Avg Grey Ratio Loss",
                 "Avg TV Loss (abs)", "Avg TV Loss (sq)")
    
    lossplotheader = list(logheader[3:])
    lossplotlegend = ["Similarity Loss", "Dissimilarity Loss",
                      "Grey Mean Loss", "Grey Ratio Loss",
                      "TV Loss (abs)", "TV Loss (sq)"]
    assert len(available_losses) == len(lossplotheader) == len(lossplotlegend)
    argsdict = vars(args)
    for i, loss in reversed(list(enumerate(available_losses))):
        if argsdict[loss] == 0:   # if loss unused (weight 0)
            del lossplotheader[i] # do not include in plot
            del lossplotlegend[i]
    
    plots = [
        ("plot_classif.pdf", {
            'x': logheader[0],
            'y': [logheader[1], logheader[2]],
            'ylim': (0, 1),
            'title': "Average Classification Score"
        }, ["Style Font", "Content Font"]),
        ("plot_loss.pdf", {
            'x': logheader[0],
            'y': lossplotheader,
            'title': "Mean Loss"
        }, lossplotlegend)
    ]
    logger = Logger(*logheader, file=logfile)
    plotter = Plotter(logfile, plots)
    atexit.register(plotter.save) # plot on exit
    
    set_random_seed(args.seed)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    classifier = Supermodel(style.models.index, args.model).to(device)
    classifier.load(args.state.format(device))
    classifier.no_grad()
    
    # Size of batches: 676 x 1 x 64 x 64
    batch_content = load_font_abcabc(args.contentfont).to(device)
    batch_style = load_font_abcabc(args.stylefont).to(device)
    # Size of fonts: 26 x 1 x 64 x 64
    font_content = load_font_abc(args.contentfont).to(device)
    font_style = load_font_abc(args.stylefont).to(device)
    font_optim = load_font_abc(args.contentfont).to(device)
    font_optim.requires_grad_(True)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD([font_optim], lr=args.lr, momentum=0.95)
    
    target_true_batch = torch.ones((config.LETTERS, 1), device=device)
    target_false_batch = torch.zeros((config.LETTERS, 1), device=device)
    
    target_greymean = torch.tensor(args.greymean, device=device)
    target_greymean_batch = target_greymean.repeat(config.LETTERS, 1)
    
    target_greyratio = torch.tensor(args.greyratio, device=device)
    target_greyratio_batch = target_greyratio.repeat(config.LETTERS, 1)
    
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)
    
    # At step zero equal to content font
    savedata(font_optim, "{:0>{len}}".format(0, len=len(str(args.steps+1))))
    
    print('Startup took {:.3f} sec'.format(time.time()-starttime))
    
    for step in range(1, args.steps+1):
        def closure():
            optimizer.zero_grad()
            
            pred_style   = classifier.model((rand_glyphs(font_style), font_optim))
            pred_content = classifier.model((rand_glyphs(font_content), font_optim))
            
            loss_sim     = criterion(pred_style, target_true_batch)     * args.LOSS_SIM
            loss_dissim  = criterion(pred_content, target_false_batch)  * args.LOSS_DISSIM
            loss_grmean  = greymean_loss(font_optim, target_greymean)   * args.LOSS_GRMEAN
            loss_grratio = greyratio_loss(font_optim, target_greyratio) * args.LOSS_GRRATIO
            loss_tvabs   = tvloss_abs(font_optim)                       * args.LOSS_TVABS
            loss_tvsq    = tvloss_sq(font_optim)                        * args.LOSS_TVSQ
            
            loss = loss_sim + loss_dissim + loss_grmean + loss_grratio + loss_tvabs + loss_tvsq
            loss.backward()
            
            with torch.no_grad():
                # For classification optim needs to be copied aabbcc
                # and style/content abcabc
                batch_optim = batch_aabbcc(font_optim)
                classif_style   = classifier.model((batch_style, batch_optim))
                classif_content = classifier.model((batch_content, batch_optim))
            
            logger.update(step,
                classif_style.mean().item(), classif_content.mean().item(),
                loss_sim.mean().item(), loss_dissim.mean().item(),
                loss_grmean.mean().item(), loss_grratio.mean().item(),
                loss_tvabs.mean().item(), loss_tvsq.mean().item()
            )
            
            print('S {:{len}}   '
                  'Sty C {:.2f}%   '
                  'Cont C {:.2f}%   '
                  'Sim L {:.{precision}f}   '
                  'Dissim L {:.{precision}f}   '
                  'Gr Mean L {:.{precision}f}   '
                  'Gr Ratio L {:.{precision}f}   '
                  'TV L (abs) {:.{precision}f}   '
                  'TV L (sq) {:.{precision}f}   '
                  'T {:.0f}'.format(step,
                      classif_style.mean().item()*100, classif_content.mean().item()*100,
                      loss_sim.mean().item(), loss_dissim.mean().item(),
                      loss_grmean.mean().item(), loss_grratio.mean().item(),
                      loss_tvabs.mean().item(), loss_tvsq.mean().item(),
                      time.time()-starttime,
                      len=len(str(args.steps+1)),
                      precision=5))
            
            return loss
        
        optimizer.step(closure)
        font_optim.data.clamp_(0, 1) # Limit to admissible values
        savedata(font_optim, "{:0>{len}}".format(step, len=len(str(args.steps+1))))

if __name__ == "__main__":
    main()
