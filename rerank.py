# -----------------------------------------------------------
# "Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking"
# WangTan, XingXu, YangYang, Alan Hanjalic, HengtaoShen, JingkuanSong
# ACM Multimedia 2019, Nice, France
# Writen by WangTan, 2019
# ------------------------------------------------------------

# The core cross-modal re-ranking module without supervision
# and can be added into other image-text matching methods

import numpy as np
import utils


def i2t_rerank(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):
            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t


def t2i_rerank(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]
            ranks = 1e20
            for k in range(5):
                tmp = np.where(query == i//5 * 5 + k)[0][0]
                if tmp < ranks:
                    ranks = tmp
            address = np.append(address, ranks)

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i


def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = input[index]
        # Score
        if index == 197:
            print('s')
        rank = 1e20
        for i in range(5 * index, min(5 * index + 5, image_size*5), 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = input[5 * index + i]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


# The accuracy computing
# Input the prediction similarity score matrix (d * 5d)
d = np.load('flickr1_stage1_d.npy')

# calculate the i2t score after rerank
sort_rerank = i2t_rerank(d, 15, 1)
(r1i, r5i, r10i, medri, meanri), _ = acc_i2t2(np.argsort(-d, 1))
(r1i2, r5i2, r10i2, medri2, meanri2), _ = acc_i2t2(sort_rerank)

print(r1i, r5i, r10i, medri, meanri)
print(r1i2, r5i2, r10i2, medri2, meanri2)


# calculate the t2i score after rerank

# sort_rerank = t2i_rerank(d, 20, 1)
# (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2(np.argsort(-d, 0))
# (r1t2, r5t2, r10t2, medrt2, meanrt2), _ = acc_t2i2(sort_rerank)
#
# print(r1t, r5t, r10t, medrt, meanrt)
# print(r1t2, r5t2, r10t2, medrt2, meanrt2)