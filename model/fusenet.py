''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from visualizer import get_local

class LayerNormNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out

        self.fc1 = nn.Linear(input_dim, hidden_dim,)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(ConvNet, self).__init__()
        self.conv1 =  nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=(1, 7), stride=(2,1), padding=(0, 3)),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),)
        self.conv2 =  nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=(1, 9), stride=(2,1), padding=(0, 4)),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),)
        self.conv3 =  nn.Sequential(nn.Conv2d(input_dim, out_dim, kernel_size=(1, 11), stride=(2,1), padding=(0, 5)),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),)

        self.mlp = LayerNormNet(out_dim*3, hidden_dim, out_dim)
    def forward(self, sequence_feature, structure_feature):
        combined_feature = torch.stack((sequence_feature, structure_feature), dim=2)
        combined_feature = combined_feature.permute(0, 3, 2, 1)
        feature1 = self.conv1(combined_feature).squeeze(2)
        feature2 = self.conv2(combined_feature).squeeze(2)
        feature3 = self.conv3(combined_feature).squeeze(2)
        fuse_feature = torch.cat([feature1, feature2, feature3], dim=1).permute(0, 2, 1)
        out = self.mlp(fuse_feature)
        return out



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    @get_local('attn')
    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.cross_attn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.cross_attn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, seq_input, struc_input, slf_attn_mask=None):
        struc_input, enc_slf_attn = self.cross_attn1(
            seq_input, struc_input, struc_input, mask=slf_attn_mask)
        struc_output = self.pos_ffn1(struc_input)

        seq_output, enc_slf_attn = self.cross_attn1(
            struc_output, seq_input, seq_input, mask=slf_attn_mask)
        seq_output = self.pos_ffn1(seq_output)
        return seq_output, struc_output, enc_slf_attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=1025):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=1025, scale_emb=False):

        super().__init__()

        self.position_enc1 = PositionalEncoding(d_model, n_position=n_position)
        self.position_enc2 = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.local_stack = nn.ModuleList([
            ConvNet(d_model, 512, d_model)
            for _ in range(n_layers)])
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, seq_input, struc_input, src_mask, return_attns=False):

        enc_slf_attn_list = []
        seq_input = self.dropout(self.position_enc1(seq_input))
        seq_input = self.layer_norm1(seq_input)
        struc_input = self.dropout(self.position_enc2(struc_input))
        struc_input = self.layer_norm2(struc_input)
        seq_input1 = seq_input.detach()
        # for enc_layer in self.layer_stack:
        for i in range(len(self.layer_stack)):
            enc_layer = self.layer_stack[i]
            local_layer = self.local_stack[i]
            seq_input, struc_input, enc_slf_attn = enc_layer(seq_input, struc_input, slf_attn_mask=src_mask)
            seq_input1 = local_layer(seq_input, seq_input1)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return seq_input, seq_input1, enc_slf_attn_list
        return seq_input, seq_input1

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, d_model=256, d_inner=512, n_layers=6, n_head=8, d_k=160, d_v=160, dropout=0.1, n_position=1025):

        super().__init__()
        self.d_model = d_model
        self.sequence_linear = LayerNormNet(1280, 512, d_model)
        self.structure_linear = LayerNormNet(1024, 512, d_model)
        self.encoder = Encoder(n_position=n_position, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, scale_emb=False)
        self.mlp1 = nn.Sequential(nn.Linear(256, 256),
                                 nn.LayerNorm(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),)
                                #  nn.LayerNorm(256),
                                #  nn.ReLU(),
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 


    def forward(self, src_seq, src_struc, src_mask):
        src_seq = self.sequence_linear(src_seq)
        src_struc = self.structure_linear(src_struc)
        enc_output, local_output, *_ = self.encoder(src_seq, src_struc, src_mask.unsqueeze(-2))
        fuse_feature = enc_output + local_output
        fuse_f = self.mlp1(fuse_feature) + fuse_feature
        masked_enc_output = fuse_f * src_mask.unsqueeze(-1)
        x = masked_enc_output.sum(dim=1) / src_mask.sum(dim=1).unsqueeze(-1)
        return x