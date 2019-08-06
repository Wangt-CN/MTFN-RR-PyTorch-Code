#encoding:utf-8
# -----------------------------------------------------------
# "Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking"
# WangTan, XingXu, YangYang, Alan Hanjalic, HengtaoShen, JingkuanSong
# ACM Multimedia 2019, Nice, France
# Writen by WangTan, 2019.  Our code is depended on MUTAN
# ------------------------------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import seq2vec
import click

import utils
import engine
import data
import model as models
from vocab import deserialize_vocab



def main():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/data/linkaiyi/scan/data/f30k_precomp',
                        help='path to datasets')
    parser.add_argument('--path_opt', default='option/FusionNoattn_baseline.yaml', type=str,
                        help='path to a yaml options file')
    parser.add_argument('--data_name', default='flickr30k_splits',
                        help='{coco,f30k}_splits')
    parser.add_argument('--logger_name', default='./log_2',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--vocab_path', default='/home/linkaiyi/fusion_wangtan/Fusion_flickr/Fusion_10.28/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='/mnt/data/linkaiyi/mscoco/fusion/Fusion_flic/runs/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--num_epochs', default=120, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr_update', default=20, type=int,
                        help='Number of epochs to update the learning rate.')


    opt = parser.parse_args()
    if os.path.isdir(opt.logger_name):
        if click.confirm('Logs directory already exists in {}. Erase?'
                                 .format(opt.logger_name, default=False)):
            os.system('rm -r ' + opt.logger_name)
    tb_logger.configure(opt.logger_name, flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    #########################################################################################
    # Create options
    #########################################################################################

    options = {
        'logs': {},
        'coco': {},
        'model': {
        'seq2vec': {}
        },
        'optim': {
        }
    }
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]
    opt.vocab_size = len(vocab)



    # Create dataset, model, criterion and optimizer

    train_loader, val_loader = data.get_loaders(
       opt.data_path, vocab, opt.batch_size, opt.workers, opt)
    model = models.factory(options['model'],
                           vocab_word,
                           cuda=True, data_parallel=False)

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 128])).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=options['optim']['lr'])

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            engine.validate(val_loader, model, criterion, optimizer, opt.batch_size)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        start_epoch = 0

    # Train the Model
    best_rsum = 0
    for epoch in range(start_epoch, opt.num_epochs):


        adjust_learning_rate(opt, options, optimizer, epoch)

        # train for one epoch
        
        engine.train(train_loader, model, criterion, optimizer, epoch, print_freq=10)
        

        # evaluate on validation set
        rsum = engine.validate(val_loader, model, criterion, optimizer, opt.batch_size)

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'baseline',
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'options': options,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}{}.pth.tar'.format(epoch, best_rsum), prefix=opt.model_name + '/')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, options, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    if epoch < 50:
        lr = options['optim']['lr']/10 * (epoch//8 + 4)
    elif epoch <100:
        lr = options['optim']['lr']
    elif epoch <160:
        lr = options['optim']['lr']/10
    # lr = options['optim']['lr'] * (0.5 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

