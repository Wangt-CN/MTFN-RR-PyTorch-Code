#encoding:utf-8
# -----------------------------------------------------------
# "Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking"
# WangTan, XingXu, YangYang, Alan Hanjalic, HengtaoShen, JingkuanSong
# ACM Multimedia 2019, Nice, France
# Writen by WangTan, 2019
# ------------------------------------------------------------

import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import seq2vec
import tensorboard_logger as tb_logger
import logging


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        images, captions, lengths, ids = train_data
        batch_size = images.size(0)
        margin = 0.2
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_text = Variable(captions)
        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_text = input_text.cuda()

        #target_answer = Variable(sample['answer'].cuda(async=True))

        # compute output and loss
        scores = model(input_visual, input_text)
        torch.cuda.synchronize()
        loss = utils.calcul_loss(scores, input_visual.size(0), margin)
        # label_ = torch.eye(batch_size).long()
        # label = Variable(label_).view(-1).cuda()
        # loss_all = criterion(sims.view(-1, 2), label)

        train_logger.update('L', loss.cpu().data.numpy())


        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        # loss=train_logger['L'], loss_p=train_logger['loss_p'], loss_n=train_logger['loss_p'],
                        # acc=train_logger['acc'], rec=train_logger['rec'], pre=train_logger['pre'],
                        elog=str(train_logger)))

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)




def validate(val_loader, model, criterion, optimizer, batch_size):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_ii = torch.zeros(5070, 36, 2048)
    input_visual = []
    input_text = []
    ids_ = []

    # input_visual = np.zeros((len(val_loader.dataset), 49, 2048))
    # input_text = np.zeros((len(val_loader.dataset), 2400))
    d = np.zeros((1014, 5070))
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data

        input_ii[ids] = images
        # input_visual.append(images)
        input_text.append(captions)
        ids_.append(ids)

    input_ii = input_ii[[i for i in range(0, 5070, 5)]]
    input_visual = [input_ii[batch_size*i:min(batch_size*(i+1), 1014)] for i in range(1014//batch_size + 1)]
    del input_ii

    for j in range(len(input_visual)):
        for k in range(len(input_text)):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (j, k))

            input_v = input_visual[j]
            input_t = input_text[k]
            batch_size_v = input_v.size(0)
            batch_size_t = input_t.size(0)
            ims = Variable(input_v).cuda()
            txs = Variable(input_t).cuda()
            sums = model(ims, txs)
            # sums = sums.view(batch_size_v, batch_size_t)

            d[batch_size*j:min(batch_size*(j+1), 1014), batch_size*k:min(batch_size*(k+1), 5070)] = sums.cpu().data.numpy()
        sys.stdout.write('\n')

    # np.save('flickr1_stage1_d', d)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = r1t + r5t + r10t + r1i + r5i + r10i

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


# for (1000, 5000)
def validate2(val_loader, model, criterion, optimizer):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 36, 2048))
    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    maxl = 0
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids, maxlength = val_data
        if maxlength > maxl:
            maxl = maxlength
        batch_size = images.size(0)
        input_visual[ids] = (images.numpy().copy())
        input_text[ids,:captions.size(1)] = (captions.numpy().copy())

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    input_visual = input_visual[:896]
    input_text = input_text[:4480]
    d = utils.shard_dis(input_visual, input_text, model)


    # sim = utils.shard_dis(input_visual, input_text, model)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = r1t + r5t + r10t + r1i + r5i + r10i

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


