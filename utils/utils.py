import csv
import math
import os
import pickle
import random
import subprocess
from re import L

import numpy as np
import torch

from .distance_map import get_dist_map

def get_reconstruction_loss_coefficient(training_step, num_total_training_steps=20000, start_MLM_coefficient=0.5, end_MLM_coefficient=0.05):
    ratio_total_steps = training_step / num_total_training_steps
    cosine_scaler = 0.5 * (1.0 + math.cos(math.pi * ratio_total_steps))
    reconstruction_loss_coeff = end_MLM_coefficient + cosine_scaler * (start_MLM_coefficient - end_MLM_coefficient)
    return reconstruction_loss_coeff

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        a = a['representations'][33]
    return a


def load_esm(lookup, esm_dir="./esm_data"):
    esm = format_esm(torch.load(os.path.join(esm_dir, lookup + ".pt")))
    return esm.unsqueeze(0)


def esm_embedding(ec_id_dict, device, dtype, esm_dir="./esm_data"):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [load_esm(id, esm_dir=esm_dir) for id in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test_ensemble(id_ec_test, device, dtype, esm_dir="./esm_data"):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id, esm_dir=esm_dir) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_name="./split100", 
                            output_dir="./esm_data"):
    print("Retriving ESM1b embedding for ", fasta_name + '.fasta')
    esm_script = "esm/scripts/extract.py"
    esm_type = "./esm/esm1b_t33_650M_UR50S.pt"
    command = ["python", esm_script, esm_type, 
              fasta_name + '.fasta', output_dir, "--include", "mean"]
    subprocess.run(command)
 
def compute_esm_distance(train_file, data_dir = "./data"):
    print("Computing ESM distance for ", data_dir + '/' + train_file + '.csv')
    ensure_dirs(os.path.join(data_dir, "distance_map"))
    _, ec_id_dict = get_ec_id_dict(os.path.join(data_dir, train_file + '.csv'))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    esm_emb = esm_embedding(ec_id_dict, device, dtype, data_dir=data_dir)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    pickle.dump(esm_dist, open(os.path.join(data_dir, "distance_map", train_file + '_esm.pkl'), 'wb'))
    pickle.dump(esm_emb, open(os.path.join(data_dir, "distance_map", train_file + '_esm_emb.pkl'), 'wb'))
    
def prepare_infer_fasta(fasta_name="./split100"):
    retrive_esm1b_embedding(fasta_name)
    csvfile = open(fasta_name + '.csv', 'r')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open(fasta_name + '.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def replacement(seq: str) -> str:
    pairs = [('A', 'V'), ('S', 'T'), ('F', 'Y'), ('K', 'R'), ('C', 'M'), ('D', 'E'), ('N', 'Q'), ('V', 'I')]
    reversed_pairs = [(b, a) for a, b in pairs]
    pair_dict = {**{k: v for k, v in pairs}, **{k: v for k, v in reversed_pairs}}
    mu, sigma = .01, .002 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1)
    new_sequence = ""
    for char in seq:
        if char in pair_dict and random.random() < s:
            new_sequence += pair_dict[char]
        else:
            new_sequence += char
    return new_sequence

def random_shuffling(sequence: str) -> str:
    alpha = random.randint(0, len(sequence) - 2)
    beta = random.randint(alpha + 2, len(sequence))
    alpha = max(1, min(alpha, len(sequence) - 1))
    beta = max(alpha + 1, min(beta, len(sequence)))
    start = sequence[:alpha]
    middle = list(sequence[alpha:beta])
    end = sequence[beta:]
    random.shuffle(middle)
    new_sequence = start + ''.join(middle) + end
    return new_sequence

def mask_sequences(ids, csv_name, fasta_name, data_dir = "./split100") :
    single_id, second_id, third_id, fourth_id, fifth_id = ids[0], ids[1], ids[2], ids[3], ids[4]
    csv_file = open(os.path.join(data_dir, csv_name + '.csv'))
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open(os.path.join(data_dir, fasta_name + '.fasta'), 'w')
    single_id, second_id, third_id, fourth_id, fifth_id = set(single_id), set(second_id), set(third_id), set(fourth_id), set(fifth_id)
    ids = [single_id, second_id, third_id, fourth_id, fifth_id]

    for i, rows in enumerate(csvreader):
        for j in range(len(ids)):
            if rows[0] in ids[j]:
                for k in range((5-j)*10):
                    seq = rows[2].strip()
                    seq = random_shuffling(replacement(seq))
                    output_fasta.write('>' + rows[0] + '_' + str(k) + '\n')
                    output_fasta.write(seq + '\n')
                for k in range((5-j)*10, (5-j)*10+10):
                    seq = rows[2].strip()
                    mu, sigma = .10, .02 # mean and standard deviation
                    s = np.random.normal(mu, sigma, 1)
                    mut_rate = s[0]
                    times = math.ceil(len(seq) * mut_rate)
                    for k in range(times):
                        position = random.randint(1 , len(seq) - 1)
                        seq = mutate(seq, position)
                    seq = seq.replace('*', '<mask>')
                    output_fasta.write('>' + rows[0] + '_' + str(k) + '\n')
                    output_fasta.write(seq + '\n')

def mutate_single_seq_ECs(train_file, data_dir = "./split100"):
    print("Mutating single-seq EC numbers for ", data_dir + '/' + train_file + '.csv')
    id_ec, ec_id =  get_ec_id_dict(os.path.join(data_dir, train_file + '.csv'))
    single_ec, second_ec, third_ec, fourth_ec, fifth_ec = set(), set(), set(), set(), set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
        if len(ec_id[ec]) == 2:
            second_ec.add(ec)
        if len(ec_id[ec]) == 3:
            third_ec.add(ec)
        if len(ec_id[ec]) == 4:
            fourth_ec.add(ec)
        if len(ec_id[ec]) == 5:
            fifth_ec.add(ec)
    single_id, second_id, third_id, fourth_id, fifth_id = set(), set(), set(), set(), set()
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                single_id.add(id)
                break
            if ec in second_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                second_id.add(id)
                break
            if ec in third_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                third_id.add(id)
                break
            if ec in fourth_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                fourth_id.add(id)
                break
            if ec in fifth_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                fifth_id.add(id)
                break
    ids = [single_id, second_id, third_id, fourth_id, fifth_id]
    print("Number of EC numbers with only one sequences:",len(single_ec),len(second_ec),len(third_ec),len(fourth_ec),len(fifth_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(single_id),len(second_id),len(third_id),len(fourth_id),len(fifth_id))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id), len(second_ec) - len(second_id), len(third_ec) - len(third_id), len(fourth_ec) - len(fourth_id), len(fifth_ec) - len(fifth_id))
    mask_sequences(ids, train_file, train_file+'_single_seq_ECs')
    fasta_name = train_file+'_single_seq_ECs'
    return fasta_name
