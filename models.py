'''
This fils builds the neural net used for beam-prediction. It is a modified code from that of resnet
offered at https://github.com/pytorch/vision/tree/master/torchvision/models
--------------------------
Main functions:
1- resnet18_mod: A modified version of resnet18. It incorporates a different fully-connected
layer to suit the beam-prediction problem. It is usually used with the pre-trained option set to true
--------------------------
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        #print("width", width)
        #print("planes", planes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.output_dim = num_classes
        print('Output layer dim = ' + str(self.output_dim))
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #print("layers ", layers)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_reduce = nn.Linear(512 * block.expansion, 128)  # Reduce dimension to 128
        #self.fc = nn.Linear(128, self.output_dim) 
        #self.fc = nn.Linear(128, self.output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        y = torch.flatten(x, 1)
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        y = self.fc_reduce(y)
        return y


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    print(block)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # print(state_dict.keys())
        # print(model.fc.weight)
        for key in kwargs:
            if key == 'num_classes':
                num_classes = kwargs[key]

        if num_classes != 1000:
            state_dict['fc_reduce.weight'] = nn.init.xavier_normal_(model.fc_reduce.weight, gain=1)
            state_dict['fc_reduce.bias'] = nn.init.constant_(model.fc_reduce.bias, val=0)
            #state_dict['fc.weight'] = nn.init.xavier_normal_(model.fc.weight, gain=1)
            #state_dict['fc.bias'] = nn.init.constant_(model.fc.bias, val=0)
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            

        model.load_state_dict(state_dict)
    return model


def resnet18_mod(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-18 model for camera-beam prediction.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
                   
                   
def resnet50(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)                   
                   
                   
def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)     

node = 128

class NN_beam_pred(nn.Module):
        def __init__(self, num_features, num_output):
            super(NN_beam_pred, self).__init__()
            
            self.layer_1 = nn.Linear(num_features, node)
            self.layer_2 = nn.Linear(node, node)
            self.layer_3 = nn.Linear(node, num_output)
            #self.layer_out = nn.Linear(node, num_output)
            self.relu = nn.ReLU()
            
            
            
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            #y = self.layer_out(x)
            return x


class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
    




# Image Feature Extractor
class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super(ImageFeatureExtractor, self).__init__()
        base_model = resnet50(pretrained=True, num_classes=64)
        #base_model.fc = nn.Identity()  # Remove classification layer
        self.feature_extractor = base_model
        #self.fc = nn.Linear(128, output_dim)  # Project to desired output dimension
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        #x = self.fc(x)
        x = self.bn(x)
        return x

# Position Feature Extractor
class PosFeatureExtractor(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super(PosFeatureExtractor, self).__init__()
        self.feature_extractor = NN_beam_pred(num_features=input_dim, num_output=output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        return x

# Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=64):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Main Model with Named Sub-Networks
class Classifier(nn.Module):
    def __init__(self, use_image=True, use_pos=True, feature_dim=128, num_classes=64):
        super(Classifier, self).__init__()
        
        # Store sub-networks in a dictionary
        self.sub_networks = nn.ModuleDict()

        if use_image:
            self.sub_networks["images"] = ImageFeatureExtractor(output_dim=feature_dim)
        if use_pos:
            self.sub_networks["pos_height"] = PosFeatureExtractor(output_dim=feature_dim)

        # Determine input size for classification head
        input_dim = (feature_dim if use_image else 0) + (feature_dim if use_pos else 0)
        self.classifier = ClassificationHead(input_dim, num_classes)

    def forward(self, images=None, pos_height=None):
        features = []

        if "images" in self.sub_networks and images is not None:
            features.append(self.sub_networks["images"](images))

        if "pos_height" in self.sub_networks and pos_height is not None:
            features.append(self.sub_networks["pos_height"](pos_height))

        if not features:
            raise ValueError("At least one modality (image or pos) must be used")

        x = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.classifier(x)
    



def make_drone_classifier(user_modalities):
    if "images" in user_modalities and "pos_height" in user_modalities:
        user_model = Classifier(use_image=True, use_pos=True)
    elif "pos_height" in user_modalities:
        user_model = Classifier(use_image=False, use_pos=True)
    elif "images" in user_modalities:
        user_model = Classifier(use_image=True, use_pos=False)

    return user_model