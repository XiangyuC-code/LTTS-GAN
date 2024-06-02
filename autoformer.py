import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, SqueezeAndExciteFusionAdd
import math
import numpy as np

class Generator(nn.Module):
    def __init__(self,configs):
        super(Generator,self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.output_attention = configs.output_attention

        # Decomp
        #kernel_size = configs.moving_avg
        #self.decomp = series_decomp(kernel_size)

        self.l1 = nn.Linear(configs.latent_dim, configs.seq_len*configs.d_model)
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.d_model, configs.dropout)

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.fusion = SqueezeAndExciteFusionAdd(channels_in=configs.c_out)

        '''
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(0.5)
        '''

    
    def forward(self, x, trend1, trend2):
        x = self.l1(x).view(-1,self.seq_len,self.d_model)
        x = self.enc_embedding(x)
        seasonal_part, trend_part1, trend_part2 = self.decoder(x, trend1=trend1, trend2=trend2)
        trend_part1, trend_part2 = trend_part1.transpose(1, 2).contiguous(), trend_part2.transpose(1, 2).contiguous()
        out = seasonal_part + self.fusion(trend_part1, trend_part2).transpose(1,2).contiguous()
        out = out.view(-1,self.c_out,1,self.seq_len)
        return out


