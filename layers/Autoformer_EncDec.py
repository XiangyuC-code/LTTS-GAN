import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=3, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv1d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool1d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    

class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        if channels_in//3 == 0:
            self.se_c1 = SqueezeAndExcitation(channels_in, reduction=1,
                                            activation=activation)
            self.se_c2 = SqueezeAndExcitation(channels_in, reduction=1,
                                                activation=activation)
        else:
            self.se_c1 = SqueezeAndExcitation(channels_in,
                                            activation=activation)
            self.se_c2 = SqueezeAndExcitation(channels_in,
                                                activation=activation)

    def forward(self, c1, c2):
        c1 = self.se_c1(c1)
        c2 = self.se_c2(c2)
        out = c1 + c2
        return out
    

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp_input(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_input, self).__init__()
        self.moving_avg0 = moving_avg(kernel_size[0], stride=1)
        self.moving_avg1 = moving_avg(kernel_size[1], stride=1)

    def forward(self, x):
        moving_mean0 = self.moving_avg0(x)
        res0 = x - moving_mean0

        moving_mean1 = self.moving_avg1(x)
        res1 = x - moving_mean1

        res = (res0 + res1) / 2
        return res, moving_mean0, moving_mean1
    

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, channels):
        super(series_decomp, self).__init__()
        self.moving_avg0 = moving_avg(kernel_size[0], stride=1)
        self.moving_avg1 = moving_avg(kernel_size[1], stride=1)
        self.x_fusion = SqueezeAndExciteFusionAdd(channels_in=channels)

    def forward(self, x):
        moving_mean0 = self.moving_avg0(x)
        res0 = x - moving_mean0

        moving_mean1 = self.moving_avg1(x)
        res1 = x - moving_mean1

        res0, res1 = res0.transpose(1, 2).contiguous(), res1.transpose(1, 2).contiguous()
        res = self.x_fusion(res0, res1).transpose(1,2).contiguous()
        return res, moving_mean0, moving_mean1

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm1d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.local1 = ConvBN(d_model,d_model,kernel_size=3)
        self.local2 = ConvBN(d_model,d_model,kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg, d_model)
        self.decomp2 = series_decomp(moving_avg, d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_mask=None):
        _, S, C = x.shape
        # local attention
        local = self.local1(x.view(-1,C,S)) + self.local2(x.view(-1,C,S))
        local = local.view(-1,S,C)

        # local + global
        x = local + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend11, trend12 = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend21, trend22 = self.decomp2(x + y)

        residual_trend1 = trend11 + trend21 
        residual_trend1 = self.projection(residual_trend1.permute(0, 2, 1)).transpose(1, 2)

        residual_trend2 = trend12 + trend22 
        residual_trend2 = self.projection(residual_trend2.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend1, residual_trend2


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, x_mask=None, trend1=None, trend2=None):
        for layer in self.layers:
            x, residual_trend1, residual_trend2 = layer(x, x_mask=x_mask)
            trend1 = trend1 + residual_trend1
            trend2 = trend2 + residual_trend2

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend1, trend2
