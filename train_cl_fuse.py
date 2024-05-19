import argparse
import os
import pickle
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

from dataset.mine_hard import mine_hard_negative
from dataset.Dataset_cl_squence import Triplet_dataset, collate_fn, valid_collate_fn
from model.fusenet import Transformer
from utils import (ensure_dirs, get_dist_map, get_ec_id_dict, seed_everything)

from utils.distance_map import *
from utils.evaluate import *
from utils.utils import *


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-n', '--model_name', type=str, default='squence-3di')
    parser.add_argument('-m', '--model_dir', type=str, default='./results/squence-3di-all_tripletsigmoid_bceloss')
    parser.add_argument('-c', '--ckp_dir', type=str, default='./results/squence-3di-all_tripletsigmoid_bceloss')

    parser.add_argument('--data_dir', type=str, default='./CLEAN')
    parser.add_argument('-t', '--training_data', type=str, default='split100')
    parser.add_argument('-v', '--valid_data', type=str, default='split100')

    parser.add_argument('--batch_size', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--adaptive_rate', type=int, default=1)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    return args

def get_train_dataloader(dist_map, id_ec, ec_id, args):
    anchors = []
    for id in id_ec.keys():
        anchors.append(id)
    params = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'drop_last': True, 'pin_memory': True}
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset(id_ec, ec_id, negative, anchors, data_dir=args.data_dir, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, **params, collate_fn=collate_fn, sampler=train_sampler)
    return train_loader

def get_valid_dataloader(dist_map, id_ec, ec_id, args):
    anchors = []
    for id in id_ec.keys():
        anchors.append(id)
    params = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    negative = mine_hard_negative(dist_map, 30)
    valid_data = Triplet_dataset(id_ec, ec_id, negative, anchors, data_dir=args.data_dir, split='valid')
    valid_loader = torch.utils.data.DataLoader(valid_data, **params, collate_fn=valid_collate_fn)
    return valid_loader

def validate(model, valid_loader, ec_id_dict, device, dtype, criterion):
    model.eval()
    total_loss = 0.
    features = {}
    for ec in list(ec_id_dict.keys()):
        features[ec] = []
    for batch, data in enumerate(valid_loader):
        with torch.no_grad():
            a_esm, p_esm, n_esm, a_3di, p_3di, n_3di, mask_esm, ec_labels = data
            anchor_out = model(a_esm.to(device=device, dtype=dtype), a_3di.to(device=device, dtype=dtype), mask_esm.to(device=device, dtype=dtype))
            positive_out = model(p_esm.to(device=device, dtype=dtype), p_3di.to(device=device, dtype=dtype), mask_esm.to(device=device, dtype=dtype))
            negative_out = model(n_esm.to(device=device, dtype=dtype), n_3di.to(device=device, dtype=dtype), mask_esm.to(device=device, dtype=dtype))
            loss = criterion(anchor_out, positive_out, negative_out)
            total_loss += loss.item()
            for i in range(len(ec_labels)):
                for ec_number in ec_labels[i]:
                    features[ec_number].append(anchor_out[i].unsqueeze(0).cpu().detach())
    esm_emb = []
    for ec in list(ec_id_dict.keys()):
        esm_to_cat = torch.cat(features[ec],dim=0)
        esm_emb.append(esm_to_cat)
    esm_emb = torch.cat(esm_emb, dim= 0)
    return esm_emb, total_loss/(batch + 1)

def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        a_esm, p_esm, n_esm, a_3di, p_3di, n_3di, mask_esm = data
        anchor_out = model(a_esm.to(device=device, dtype=dtype), a_3di.to(device=device, dtype=dtype), mask_esm.to(device=device, dtype=dtype))
        positive_out = model(p_esm.to(device=device, dtype=dtype), p_3di.to(device=device, dtype=dtype), mask_esm.to(device=device, dtype=dtype))
        negative_out = model(n_esm.to(device=device, dtype=dtype), n_3di.to(device=device, dtype=dtype), mask_esm.to(device=device, dtype=dtype))
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training losscriterion_ce(fp, labels[:,:251].to(fp.device, dtype=dtype))
    return total_loss/(batch + 1)


def main():
    seed_everything()
    args = parse()
    ensure_dirs(args.model_dir)
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_start_method('spawn')
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200000))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(device_id)
    torch.cuda.set_device(args.local_rank)
    #======================== Get Data  ===================#
    id_ec_train, ec_id_dict_train = get_ec_id_dict(os.path.join('./data', args.training_data + '.csv'))
    ec_id_train = {key: list(ec_id_dict_train[key]) for key in ec_id_dict_train.keys()}
    #======================== override args ====================#
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ', dtype, "\n==> args:", args)
    
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
    dist_map = pickle.load(
        open(os.path.join(args.data_dir,'distance_map',args.training_data + '.pkl'), 'rb'))
    #======================== initialize model =================#
    model = Transformer(d_model=256, d_inner=512, n_layers=2, n_head=8, d_k=32, d_v=32, dropout=0.1, n_position=1022)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,broadcast_buffers=False,
                                             find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    best_valid_loss = float('inf')
    # dataloader
    train_loader = get_train_dataloader(dist_map, id_ec_train, ec_id_train, args)
    valid_loader = get_valid_dataloader(dist_map, id_ec_train, ec_id_train, args)      

    print("The number of unique EC numbers: ", len(dist_map.keys()))

    #======================== training =======-=================#
    for epoch in range(1, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            if dist.get_rank() == 0:
                time1 = time.time()
                esm_emb, valid_loss = validate(model, valid_loader, ec_id_dict_train, device, dtype, criterion)
                torch.save(model.state_dict(), os.path.join(args.model_dir, model_name + '_valid_'+ str(epoch) + '_' +str(valid_loss) + '.pth'))
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    print(f'Valid from epoch : {epoch:3d}; loss: {valid_loss:6.4f}')   
                pickle.dump(esm_emb, open(os.path.join("./distance_map", args.model_name + '_train.pkl'), 'wb'))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
                # save updated model
                torch.save(model.state_dict(), os.path.join(args.model_dir, model_name + '_' + str(epoch) + '.pth'))
                # sample new distance map
                dist_map = get_dist_map(ec_id_dict_train, esm_emb, device, dtype)
                pickle.dump(dist_map, open(os.path.join("./distance_map", args.model_name + '_train_distmap.pkl'), 'wb'))
                time2 = time.time()
                print(time2- time1)
            train_loader = get_train_dataloader(dist_map, id_ec_train, ec_id_train, args)

        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion)
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, model_name + '_train_' + str(epoch) + '.pth'))
            if epoch > 5:
                os.remove(os.path.join(args.model_dir, model_name + '_train_' + str(epoch-5) + '.pth'))
        if train_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(args.model_dir, model_name + '.pth'))
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)

    torch.save(model.state_dict(), os.path.join(args.model_dir, model_name + '_final.pth'))


if __name__ == '__main__':
    main()
