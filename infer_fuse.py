import torch
from utils.utils import * 
from utils.distance_map import *
from utils.evaluate import *
import pandas as pd
import warnings

from model.fusenet import Transformer

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def model_embedding_train(id_ec_train, ec_id_dict_train, model, device, dtype, model_name):
    features = {}
    ensure_dirs('./infer/' + model_name)
    for ec in list(ec_id_dict_train.keys()):
        features[ec] = []
    print('begin')
    for id in id_ec_train.keys():
        anchor = torch.load('./esm_data_per_tok/' + id + '.pt')
        anchor = format_esm(anchor)
        anchor_3di = torch.load('./3Di_new/' + id + '.pt')
        ec_labels = id_ec_train[id]
        mask_esm = torch.ones(anchor.shape[0])
        anchor_out, _ = model(anchor.unsqueeze(0).to(device=device, dtype=dtype), anchor_3di.unsqueeze(0).to(device=device, dtype=dtype), mask_esm.unsqueeze(0).to(device=device, dtype=dtype))
        torch.save(anchor_out.cpu().detach(), './infer/'+ model_name +'/'+ id + '.pt')
        for ec_number in ec_labels:
            features[ec_number].append(anchor_out.cpu().detach())
    esm_emb = []
    for ec in list(ec_id_dict_train.keys()):
        esm_to_cat = torch.cat(features[ec],dim=0)
        esm_emb.append(esm_to_cat)
    emb_train = torch.cat(esm_emb, dim=0)
    print('finish')
    return emb_train


def model_embedding_test(id_ec_test, test_data, model, device, dtype, model_name):
    esm_emb = []
    if test_data == 'new_2023':
        test_data = 'new_2022'
    for id in id_ec_test.keys():
        anchor = torch.load('./data/'+ test_data + '/esm/' + id + '.pt')
        anchor = format_esm(anchor)
        anchor_3di = torch.load('./data/'+ test_data + '/3di/' + id + '.pt')
        mask_esm = torch.ones(anchor.shape[0])
        anchor_out, _ = model(anchor.unsqueeze(0).to(device=device, dtype=dtype), anchor_3di.unsqueeze(0).to(device=device, dtype=dtype), mask_esm.unsqueeze(0).to(device=device, dtype=dtype))
        torch.save(anchor_out.cpu().detach(), './infer/' + model_name + '/' + id + '.pt')
        esm_emb.append(anchor_out.cpu().detach())
    emb_test = torch.cat(esm_emb, dim= 0)
    return emb_test


def infer_maxsep(train_data, test_data, report_metrics = False, model_name=None, gmm = None, mode='train'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    model = Transformer(d_model=256, d_inner=512, n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.1, n_position=1022).to(device)

    try:
        checkpoint = torch.load('./results/fuse/'+ model_name +'.pth', map_location=device)
        weights_dict = {}
        for k, v in checkpoint.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
    except FileNotFoundError as error:
        raise Exception('No model found!')
            
    model.load_state_dict(weights_dict)
    model.eval()
    if mode == 'train':
        emb_train = model_embedding_train(id_ec_train, ec_id_dict_train, model, device, dtype, model_name)
        emb_train = emb_train.cpu()
        pickle.dump(emb_train, open(os.path.join("./results/fuse/", model_name + '_emb.pkl'), 'wb'))
    else:
        emb_train = pickle.load(open('./results/fuse/' + model_name + '_emb.pkl', 'rb'))
    emb_test = model_embedding_test(id_ec_test, test_data, model, device, dtype, model_name).cpu()
    device = torch.device("cpu")
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype, model_name)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/fuse/" +  test_data + '_' + model_name
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, acc = get_eval_metrics(
            pred_label, true_label, all_label)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3}')
        print('-' * 75)



if __name__ == "__main__":
    train_data = "split100"
    test_data = "new"
    infer_maxsep(train_data, test_data, report_metrics=True, model_name='fuse', mode='train')