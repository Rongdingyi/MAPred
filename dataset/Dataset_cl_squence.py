import os
import torch
import random
from utils.utils import format_esm
import pandas as pd
import csv
from dataset.mine_hard import mine_negative, random_positive
import numpy as np
import torch.nn.functional as F
import time 
import json


def batch_mask(B, N, max_length, batch, index):
    X = torch.zeros(B, max_length, N)
    for i, b in enumerate(batch):
        x = b[index]
        x_pad = F.pad(x, (0, 0, 0, max_length-len(b[index])), mode='constant', value=np.nan)
        X[i,:,:] = x_pad
    # mask = np.isnan(X)
    mask = torch.isnan(X)
    X[mask] = 0.
    mask = (torch.sum(mask, dim=2) / N).to(dtype=torch.int32)
    return X.to(dtype=torch.float32), (torch.ones_like(mask)-mask).to(dtype=torch.float32)

def collate_fn(batch):
    B = len(batch)
    # lengths = [len(b[0]) for b in batch]
    max_length = 1022
    batch0, mask0 = batch_mask(B, 1280, max_length, batch, 0)
    batch1, mask1 = batch_mask(B, 1280, max_length, batch, 1)
    batch2, mask2 = batch_mask(B, 1280, max_length, batch, 2)
    batch3, mask2 = batch_mask(B, 1024, max_length, batch, 3)
    batch4, mask2 = batch_mask(B, 1024, max_length, batch, 4)
    batch5, mask2 = batch_mask(B, 1024, max_length, batch, 5)
    return batch0, batch1, batch2, batch3, batch4, batch5, mask0

def valid_collate_fn(batch):
    B = len(batch)
    # lengths = [len(b[0]) for b in batch]
    max_length = 1022
    batch0, mask0 = batch_mask(B, 1280, max_length, batch, 0)
    batch1, mask1 = batch_mask(B, 1280, max_length, batch, 1)
    batch2, mask2 = batch_mask(B, 1280, max_length, batch, 2)
    batch3, mask2 = batch_mask(B, 1024, max_length, batch, 3)
    batch4, mask2 = batch_mask(B, 1024, max_length, batch, 4)
    batch5, mask2 = batch_mask(B, 1024, max_length, batch, 5)
    ec_lables = []
    for i, b in enumerate(batch):
        ec_lables.append(b[6])
    return batch0, batch1, batch2, batch3, batch4, batch5, mask0, ec_lables

def load_data(data_dir, anchor, pos, neg):
    a_esm = torch.load(data_dir + '/esm_data_per_tok/' + anchor + '.pt')
    if os.path.exists(data_dir + '/esm_data_per_tok_aug/' + pos + '.pt'):
        p_esm = torch.load(data_dir + '/esm_data_per_tok_aug/' + pos + '.pt')
    elif os.path.exists(data_dir + '/esm_data_per_tok/' + pos + '.pt'):
        p_esm = torch.load(data_dir + '/esm_data_per_tok/' + pos + '.pt')
    else:
        p_esm = torch.load(data_dir + '/esm_data_per_tok/' + anchor + '.pt')
    n_esm = torch.load(data_dir + '/esm_data_per_tok/' + neg + '.pt')
    a_3di = torch.load(data_dir + '/3Di_new/' + anchor + '.pt')
    if os.path.exists(data_dir + '/3Di_1_new/' + pos + '.pt'):
        p_3di = torch.load(data_dir + '/3Di_1_new/' + pos + '.pt')
    elif os.path.exists(data_dir + '/3Di_new/' + pos + '.pt'):
        p_3di = torch.load(data_dir + '/3Di_new/' + pos + '.pt')
    else:
        p_3di = torch.load(data_dir + '/3Di_new/' + anchor + '.pt')
    n_3di = torch.load(data_dir + '/3Di_new/' + neg + '.pt')
    return format_esm(a_esm), format_esm(p_esm), format_esm(n_esm), format_esm(a_3di), format_esm(p_3di), format_esm(n_3di)

class Triplet_dataset(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, anchors, data_dir='./CLEAN', split='train'):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        self.data_dir = data_dir
        self.anchors = anchors
        self.split = split

        for ec in ec_id.keys():
            if '-' not in ec:
                if len(ec_id[ec]) <= 5:
                    for i in range(len(ec_id[ec])):
                        self.full_list.append(ec)
                else:
                    for i in range(5):
                        self.full_list.append(ec)

    def __len__(self):
        if self.split == 'train':
            return len(self.full_list)
        else:
            return len(self.anchors)

    def __getitem__(self, index):
        if self.split == 'train':
            anchor_ec = self.full_list[index]
            anchor = random.choice(self.ec_id[anchor_ec])
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            a_esm, p_esm, n_esm, a_3di, p_3di, n_3di = load_data(self.data_dir, anchor, pos, neg)
            return a_esm, p_esm, n_esm, a_3di, p_3di, n_3di
        else:
            anchor = self.anchors[index]
            ec_number = self.id_ec[anchor]
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            a_esm, p_esm, n_esm, a_3di, p_3di, n_3di = load_data(self.data_dir, anchor, pos, neg)
            return a_esm, p_esm, n_esm, a_3di, p_3di, n_3di, ec_number


class Valid_dataset(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, anchors, data_dir='./CLEAN',
                 training_data='split100'):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.mine_neg = mine_neg
        self.data_dir = data_dir
        self.id_seq_a  = {}
        self.id_seq  = {}
        self.training_data = training_data
        self.anchors = anchors

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = self.anchors[index]
        ec_number = self.id_ec[anchor]
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = torch.load(self.data_dir + '/esm_data_per_tok/' + anchor + '.pt')
        if os.path.exists(self.data_dir + '/esm_data_per_tok_aug/' + pos + '.pt'):
            p = torch.load(self.data_dir + '/esm_data_per_tok_aug/' + pos + '.pt')
        elif os.path.exists(self.data_dir + '/esm_data_per_tok/' + pos + '.pt'):
            p = torch.load(self.data_dir + '/esm_data_per_tok/' + pos + '.pt')
        else:
            p = torch.load(self.data_dir + '/esm_data_per_tok/' + anchor + '.pt')
        n = torch.load(self.data_dir + '/esm_data_per_tok/' + neg + '.pt')
        return format_esm(a), format_esm(p), format_esm(n), ec_number