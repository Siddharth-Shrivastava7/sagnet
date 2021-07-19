import torch.nn as nn
affine_par = True
from torchsummary import summary
import torch 

from randomizations import StyleRandomization, ContentRandomization 


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

    def forward(self, x):
        out = self.conv2d_list[0](x)
        # print(len(self.conv2d_list)) #4 
        # print(self.conv2d_list)
        # print('*********')
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, drop=0, sagnet=True, style_stage=3):
        
        super(ResNetMulti, self).__init__() 
        self.drop = drop ## not applying currently...
        self.sagnet = sagnet
        self.style_stage = style_stage 

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # / 2 
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        ## these two last layers are for the prediction purpose..i.e. to bring the number of channels to numclasses and applying atrous ; padding series ...so i think there is no use to bring in the style net ..but while loading you need to do that...so i need to that i think  
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes) # not making use of this layer..since its result is not used 
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.dropout = nn.Dropout(self.drop)

        if self.sagnet: 
            self.style_randomization = StyleRandomization()
            self.content_randomization = ContentRandomization()

            style_layers = []
            if style_stage == 1: 
                self.in_planes = 64
                style_layers += [self._make_layer(block, 64, layers[0])]
            if style_stage <= 2:
                self.inplanes = 64 * block.expansion ## planes * block.expansion
                style_layers += [self._make_layer(block, 128, layers[1], stride=2)]
            if style_stage <=3: 
                self.inplanes = 128 * block.expansion
                style_layers+=[self._make_layer(block, 256, layers[2], stride=1, dilation=2)]
            if style_stage <=4:
                self.inplanes = 256*block.expansion
                style_layers += [self._make_layer(block, 512, layers[3], stride=1, dilation=4)]

            self.style_net = nn.Sequential(*style_layers)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        # print(self.inplanes != planes * block.expansion) ## True in each 
        # print('***************')
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        # print(downsample._modules) # ordered dict having conv2d as well...
        # print('***************')
        for i in downsample._modules['1'].parameters():
            # print(i)  # conv2d ..and batch norm 
            # print('*************')
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            if self.sagnet and i + 1 == self.style_stage: 
                x_style = self.content_randomization(x)
                x = self.style_randomization(x)
            x = layer(x)
    
        ## making up prediction layer for content (Style randomisation)
        y = self.layer6(x)

        if self.sagnet:
            ## making up pred layer for style(Content randomisation)
            x_style = self.style_net(x_style)
            y_style = self.layer6(x_style)
        else: 
            y_style = None
        # print(y.shape) # torch.Size([2, 19, 68, 121]) [N, C, H, W]
        # print(y_style.shape) # torch.Size([2, 19, 68, 121])
        return y, y_style


def sag_deeplab(pretrained=False, **kwargs): 
    model = ResNetMulti(Bottleneck,[3, 4, 23, 3], **kwargs)
    
    if pretrained:
        load_path = '/home/sidd_s/scratch/saved_models_hpc/saved_models/DANNet/trained_models/dannet_deeplab.pth' 
        print('load pretrained model from:',load_path)  
        saved_state_dict = torch.load(load_path) 
        model.load_state_dict(saved_state_dict, strict=False)
        if model.sagnet:
            states_style = {}
            for i in range(model.style_stage,5): 
                for k,v in saved_state_dict.items():
                    if k.startswith('layer' + str(i)):
                        states_style[str(i - model.style_stage) + k[6:]] = v  
            model.style_net.load_state_dict(states_style)
    return model

model = sag_deeplab(True, num_classes=19,sagnet=True, style_stage=3).cuda()
summary(model,(3,540,960))