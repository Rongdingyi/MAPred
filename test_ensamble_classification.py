import argparse
import os
import pickle
import time
import torch.distributed as dist
import json

import pandas as pd
import torch
import torch.nn as nn
from dataset.mine_hard import mine_hard_negative
from model.Classification_model import PredictNet
from dataset.Classification_dataset import Classification_dataset
from utils import (ensure_dirs, get_dist_map, get_ec_id_dict, seed_everything)
import warnings
from utils.distance_map import *
from utils.evaluate import *
from utils.utils import *

def warn(*args, **kwargs):
    pass
warnings.warn = warn


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('-c', '--csv_name', type=str, default='./results/metric_result/result.csv')
    parser.add_argument('-sm', '--ensamble_number', type=int, default=1)
    parser.add_argument('-n', '--model_path', nargs='+')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data/price')
    parser.add_argument('-t', '--test_data', type=str, default='price')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    return args

def get_test_dataloader(id_ec, ec_id, args):
    anchors = []
    for id in id_ec.keys():
        anchors.append(id)
    params = {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    train_data = Classification_dataset(id_ec, ec_id, anchors)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader

def test(model, args, valid_loader, device, dtype):
    model.eval()
    ckp_paths = args.model_path
    ids, accs, precisions, recalls, f1s = [], [], [], [], []
    ec_numbers, ec_probs, gt_numbers = [], [], []
    pred = []
    with open('./data/four_int_to_ec_number.json', 'r') as c:
        int_ec = json.load(c)
    for batch, data in enumerate(valid_loader):
        anchor, first_label, second_label, third_label, fourth_label = data
        fps, sps, tdps, thps = [], [], [], []
        ec_number, ec_prob, gt_number = [], [], []
        for i in range(args.ensamble_number):
            ckp_path = ckp_paths[i]
            model.load_state_dict(torch.load(ckp_path,map_location=torch.device('cpu')))   
            model.eval()
            fp, sp, tdp, thp = model(anchor.to(device=device, dtype=dtype))
            thp = torch.sigmoid(thp).cpu().detach().numpy()
            thps.append(thp)
        thp = np.mean(thps, axis=0)
        pred.append(thp)
    return pred

def calculate(pred, threshold):
    with open('./data/four_int_to_ec_number.json', 'r') as c:
        int_ec = json.load(c)
    ec_numbers, ec_probs, gt_numbers = [], [], []
    for thp in pred:
        ec_number, ec_prob, gt_number = [], [], []
        thp_preds = (thp >= threshold).astype(int)

        if np.sum(thp_preds) == 0:
            max_number = np.argmax(thp)
            thp_preds[:,max_number] = 1
        indices = np.where(thp_preds[0,:] == 1)
        # print(len(indices))
        for i in range(len(indices[0])):
            ec_number.append(int_ec[str(indices[0][i])])
            ec_prob.append(0.5)
        ec_numbers.append(ec_number)
        ec_probs.append(torch.tensor(ec_prob))
    return ec_numbers, ec_probs

def main():
    seed_everything()
    args = parse()
    torch.backends.cudnn.benchmark = True

    #======================== Get Data  ===================#
    id_ec, ec_id_dict = get_ec_id_dict(os.path.join('./data', args.test_data + '.csv'))
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}

    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    dtype = torch.float32
    print('==> device used:', device, '| dtype used: ', dtype, "\n==> args:", args)
    print("############ EC calling results using maximum separation ############")
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
    
    #======================== initialize model =================#
    model = PredictNet(256, drop_out=0.1).to(device).to(device)
    model = model.to(device)
    test_loader = get_test_dataloader(id_ec, ec_id, args) 
    pred = test(model, args, test_loader, device, dtype)
    thresholds = np.arange(0.6, 0.8, 0.01)
    for threshold in thresholds:
        pred_label, pred_probs = calculate(pred, threshold)
        # print(pred_probs)
        true_label, all_label = get_true_labels('./data/' + args.test_data)

        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)

        # print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> threshold: {threshold:.2} |total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)


if __name__ == '__main__':
    main()
