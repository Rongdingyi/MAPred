import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictNet(nn.Module):
    def __init__(self, input_dim, drop_out=0.1):
        super(PredictNet, self).__init__()
        self.drop_out = drop_out 
        self.first_label_block = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 7))    
        self.second_label_block = nn.Sequential(
                    nn.Linear(input_dim+7, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 72))    
        self.third_label_block = nn.Sequential(
                    nn.Linear(input_dim+72, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 253))    
        self.four_label_block = nn.Sequential(
                    nn.Linear(input_dim+253, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, 5242))     
    def forward(self, x):
        first_predict = self.first_label_block(x)
        second_predict = self.second_label_block(torch.cat((x, first_predict), dim=-1))
        third_predict = self.third_label_block(torch.cat((x, second_predict), dim=-1))
        fourth_predict = self.four_label_block(torch.cat((x, third_predict), dim=-1))
        return first_predict, second_predict, third_predict, fourth_predict