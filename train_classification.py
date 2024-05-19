import torch
import time
import os
import pickle
import json

import torch.nn as nn
import argparse
from model.losses import sigmoid_focal_loss
from model.Classification_model import PredictNet
from dataset.Classification_dataset import Classification_dataset
from utils import (ensure_dirs, seed_everything)
from utils.distance_map import *
from utils.evaluate import *
from utils.utils import *

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-v', '--valid_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-c', '--ckp_dir', type=str, default='./results/squence-3di-all__tripletsigmoid_bceloss')
    args = parser.parse_args()
    return args

def get_dataloader(id_ec, ec_id):
    params = {
        'batch_size': 50000,
        'num_workers': 32,
        'shuffle': True,
    }
    anchors = []
    for id in id_ec.keys():
        anchors.append(id)
    train_data = Classification_dataset(id_ec, ec_id, anchors)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader

def get_ec_id_dict1(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id,ec_id1,ec_id2,ec_id3,ec_id4 = {},{},{},{},{}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                ecs = ec.split('.')
                if ecs[0] not in ec_id1.keys():
                    ec_id1[ecs[0]] = set()
                    ec_id1[ecs[0]].add(rows[0])  
                else:
                    ec_id1[ecs[0]].add(rows[0])     
                if ecs[0]+'.'+ecs[1] not in ec_id2.keys():
                    ec_id2[ecs[0]+'.'+ecs[1]] = set()
                    ec_id2[ecs[0]+'.'+ecs[1]].add(rows[0])
                else:  
                    ec_id2[ecs[0]+'.'+ecs[1]].add(rows[0]) 
                if ecs[0]+'.'+ecs[1]+'.'+ecs[2] not in ec_id3.keys():
                    ec_id3[ecs[0]+'.'+ecs[1]+'.'+ecs[2]] = set()
                    ec_id3[ecs[0]+'.'+ecs[1]+'.'+ecs[2]].add(rows[0])
                else:  
                    ec_id3[ecs[0]+'.'+ecs[1]+'.'+ecs[2]].add(rows[0])            
                if ec not in ec_id4.keys():
                    ec_id4[ec] = set()
                    ec_id4[ec].add(rows[0])
                else:
                    ec_id4[ec].add(rows[0])
    return id_ec, ec_id1, ec_id2, ec_id3, ec_id4

def weights(label_dict, id_ec_train, ec_id_dict, device):
    class_weight = [0 for _ in range(len(label_dict))]
    pos_weight = [0 for _ in range(len(label_dict))]
    total_samples = len(id_ec_train)
    for ec in ec_id_dict:
        class_weight[label_dict[ec]] = 1 -len(ec_id_dict[ec]) / total_samples
        pos_weight[label_dict[ec]] = total_samples / len(ec_id_dict[ec]) - 1
    pos_weight = torch.FloatTensor(pos_weight).to(device)
    class_weight = torch.FloatTensor(class_weight).to(device)
    return class_weight, pos_weight

def valid(model, train_loader, device, dtype, class1, pose1, class2, pose2, class3, pose3, class4, pose4):
    model.eval()
    total_loss = 0.
    alpha = 0.25
    gamma = 2
    with torch.no_grad():
        for batch, data in enumerate(train_loader):
            anchor, first_label, second_label, third_label, fourth_label = data
            fp, sp, tdp, thp = model(anchor.to(device=device, dtype=dtype))
            loss = sigmoid_focal_loss(thp, fourth_label.to(thp.device, dtype=dtype),class4, pose4)
            total_loss += loss.item()
    return total_loss/(batch + 1)

def train(model, args, epoch, train_loader,
          optimizer, step_schedule, device, dtype, class1, pose1, class2, pose2, class3, pose3, class4, pose4):
    model.train()
    total_loss = 0.
    start_time = time.time()
    alpha = 0.25
    gamma = 2
    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, first_label, second_label, third_label, fourth_label = data
        fp, sp, tdp, thp = model(anchor.to(device=device, dtype=dtype))
        loss = sigmoid_focal_loss(thp, fourth_label.to(thp.device, dtype=dtype),class4, pose4)
        loss.backward()
        optimizer.step()
        step_schedule.step()

        total_loss += loss.item()
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)

# def weights():
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=15, verbose=True, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    id_ec_train, ec_id_dict_train1, ec_id_dict_train2, ec_id_dict_train3, ec_id_dict_train = get_ec_id_dict1('./data/' + args.training_data + '.csv')
    ec_id_train = {key: list(ec_id_dict_train[key]) for key in ec_id_dict_train.keys()}

    id_ec_valid, ec_id_dict_valid1, ec_id_dict_valid2, ec_id_dict_valid3, ec_id_dict_valid = get_ec_id_dict1('./data/' + args.valid_data + '.csv')
    ec_id_valid = {key: list(ec_id_dict_valid[key]) for key in ec_id_dict_valid.keys()}
    
    with open('./data/ec_number_one.json', 'r') as c:
        first_label_dict = json.load(c)
    with open('./data/ec_number_two.json', 'r') as c:
        two_label_dict = json.load(c)
    with open('./data/ec_number_three.json', 'r') as c:
        three_label_dict = json.load(c)
    with open('./data/ec_number_four.json', 'r') as c:
        four_label_dict = json.load(c)
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)

    class1, pose1 = weights(first_label_dict, id_ec_train, ec_id_dict_train1, device)
    class2, pose2 = weights(two_label_dict, id_ec_train, ec_id_dict_train2, device)
    class3, pose3 = weights(three_label_dict, id_ec_train, ec_id_dict_train3, device)
    class4, pose4 = weights(four_label_dict, id_ec_train, ec_id_dict_train, device)
    #======================== initialize model =================#

    model = PredictNet(256, drop_out=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=10, gamma=0.5, optimizer=optimizer)
    best_loss = float('inf')
    train_loader = get_dataloader(id_ec_train, ec_id_train)
    valid_loader = get_dataloader(id_ec_valid, ec_id_valid)
    print("The number of unique EC numbers: ", len(ec_id_dict_train.keys()))
    #======================== training =======-=================#
    # training
    early_stopping = EarlyStopping(os.path.join('./classification_results/', model_name + '_valid.pth'))
    for epoch in range(1, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            time1 = time.time()
            valid_loss = valid(model, valid_loader, device, dtype, class1, pose1, class2, pose2, class3, pose3, class4, pose4)
            # valid_loss = 0.1
            time2 = time.time()
            print(time2 - time1)
            early_stopping(valid_loss, model)
        # -------------------------------------------------------------------- #
        if early_stopping.early_stop:
            print("Early stopping")
            break
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, step_schedule, device, dtype, class1, pose1, class2, pose2, class3, pose3, class4, pose4)

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)


if __name__ == '__main__':
    main()
