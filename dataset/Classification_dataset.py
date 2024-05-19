import torch
from utils.utils import format_esm
import time 
import json

class Classification_dataset(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, anchors, data_dir='./CLEAN',split='train'):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = []
        self.data_dir = data_dir
        self.id_seq_a  = {}
        self.id_seq  = {}
        self.anchors = anchors
        with open('./data/ec_number_two.json', 'r') as a:
            self.two_label_dict = json.load(a)
        with open('./data/ec_number_three.json', 'r') as b:
            self.three_label_dict = json.load(b)
        with open('./data/ec_number_four.json', 'r') as c:
            self.four_label_dict = json.load(c)
        n_label = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13','n14','n15','n16']
        self.n_label_dict = {label: i for i, label in enumerate(n_label, 430)}

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = self.anchors[index]
        first_label, second_label, third_label, fourth_label = torch.zeros(7), torch.zeros(72), torch.zeros(253), torch.zeros(5242)
        for label in self.id_ec[anchor]:
            label_number = label.split('.')
            first_label[int(label_number[0])-1] = 1
            second_label[int(self.two_label_dict[label_number[0]+'.'+label_number[1]])] = 1
            third_label[int(self.three_label_dict[label_number[0]+'.'+label_number[1]+'.'+label_number[2]])] = 1
            fourth_label[int(self.four_label_dict[label_number[0]+'.'+label_number[1]+'.'+label_number[2]+'.'+label_number[3]])] = 1

        a = torch.load('./data/esm_data/' + anchor + '.pt')
        return format_esm(a), first_label, second_label, third_label, fourth_label