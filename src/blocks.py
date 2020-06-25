# Author - Dhaval Shah
# Created - 05.05.2020
# Modified - 18.05.2020

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)

        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        # x = F.elu(x)

        return x


class AnalysisBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pooling=2):
        super().__init__()
        init_out_channels = 32
        num_conv_layers = 2
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth):
            out_channels = (2**depth) * init_out_channels

            for layer in range(num_conv_layers):
                conv_layer = Conv(in_channels, out_channels)
                self.module_dict["conv_{}_{}".format(depth, layer)] = conv_layer
                in_channels, out_channels = out_channels, out_channels*2

            if depth != model_depth - 1:
                max_pool = nn.MaxPool3d(kernel_size=pooling, stride=2, padding=0)
                self.module_dict["max_pool_{}".format(depth)] = max_pool

    def forward(self, x):
        # TODO: Maybe change to tensor instead of list
        synthesis_features = []
        for key, layer in self.module_dict.items():
            x = layer(x).contiguous()

            if key.startswith("conv") and key.endswith("1"):
                synthesis_features.append(x)

        return x, synthesis_features


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        # TODO Up Convolution or Conv Transpose?
        self.up_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        return self.up_conv(x)


class SynthesisBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super().__init__()
        init_out_channels = 128
        num_conv_layers = 2
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            channels = (2**depth) * init_out_channels
            up_conv = UpConv(in_channels=channels, out_channels=channels)
            self.module_dict["deconv_{}".format(depth)] = up_conv

            for layer in range(num_conv_layers):
                in_channels, feat_channels = channels//2, channels//2
                if layer == 0:
                    in_channels = in_channels + channels

                conv_layer = Conv(in_channels=in_channels, out_channels=feat_channels)
                self.module_dict["conv_{}_{}".format(depth, layer)] = conv_layer

            if depth == 0:
                # TODO Figure out kernel size + padding + stride for final
                # 1 x 1 x 1 conv
                final_conv = nn.Conv3d(in_channels=feat_channels, out_channels=out_channels, kernel_size=1, padding=2)
                self.module_dict["final_conv"] = final_conv

    def forward(self, x, high_res_features):
        for key, layer in self.module_dict.items():
            if key.startswith("deconv"):
                x = layer(x)
                # Enforce same size
                features = high_res_features[int(key[-1])][:, :, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]].contiguous()
                x = torch.cat((features, x), dim=1).contiguous()
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=4, pooling=2, final_activation="softmax"):
        super().__init__()
        self.encoder = AnalysisBlock(in_channels=in_channels, model_depth=model_depth, pooling=pooling)
        self.encoder.cuda()
        self.decoder = SynthesisBlock(out_channels=out_channels, model_depth=model_depth)
        self.decoder.cuda()
        
        if final_activation=="softmax":
            self.final = nn.Softmax(dim=1)
            self.final.cuda()
        elif final_activation=="sigmoid":
            self.final = nn.Sigmoid()
            self.final.cuda()
        else:
            self.final = None
        #TODO other final layers
    
    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        if self.final:
            x = self.final(x)
        
        return x

if __name__ == "__main__":
    x_test = torch.randn(1, 3, 132, 132, 116)
    x_test = x_test.cuda()

    print("The shape of input: ", x_test.shape)

    encoder = AnalysisBlock(in_channels=3)
    encoder.cuda()
    x_test, h = encoder(x_test)

    decoder = SynthesisBlock(out_channels=3)
    decoder.cuda()
    y = decoder(x_test, h)

    print("The shape of output: ", y.shape)

