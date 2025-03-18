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
import torch.nn.functional as F

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
        self.fc = nn.Linear(128, output_dim)  # Project to desired output dimension
        #self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        #x = self.bn(x)
        return x

# Position Feature Extractor
class PosFeatureExtractor(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super(PosFeatureExtractor, self).__init__()
        self.feature_extractor = NN_beam_pred(num_features=input_dim, num_output=output_dim)
        #self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        #x = self.bn(x)
        return x

class MultiModalNetwork(nn.Module):
    def __init__(self, input_size_audio=None, input_size_visual=None, hidden_size=256, output_size=64, z_dim=100):
        super(MultiModalNetwork, self).__init__()
        
        self.has_audio = input_size_audio is not None
        self.has_visual = input_size_visual is not None
        self.z_dim = z_dim  # Dimensionality of random noise for generator
        
        if self.has_audio:
            # Audio feature extractor
            self.audio_feature_extractor = PosFeatureExtractor(output_dim=hidden_size)
            
            # Common and Specific classifiers for audio
            self.audio_common_classifier = nn.Linear(hidden_size // 2, output_size)
            self.audio_specific_classifier = nn.Linear(hidden_size // 2, output_size)
        
        if self.has_visual:
            # Visual feature extractor
            self.visual_feature_extractor = ImageFeatureExtractor(output_dim=hidden_size)

            # Common and Specific classifiers for visual
            self.visual_common_classifier = nn.Linear(hidden_size // 2, output_size)
            self.visual_specific_classifier = nn.Linear(hidden_size // 2, output_size)
        
        # Common classifier shared by both modalities
        self.common_classifier = nn.Linear(hidden_size // 2, output_size)  # Common features

        # Generator Network for learning modality-common features
        self.generator = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),  # Output common modality features
        )

    def forward(self, audio_input=None, visual_input=None, z=None):
        audio_features = self.audio_feature_extractor(audio_input) if self.has_audio and audio_input is not None else None
        visual_features = self.visual_feature_extractor(visual_input) if self.has_visual and visual_input is not None else None

        common_audio_features = None
        common_visual_features = None
        specific_audio_features = None
        specific_visual_features = None
        
        if self.has_audio and audio_features is not None:
            # Split audio features into common and specific parts
            common_audio_features = audio_features[:, :audio_features.size(1) // 2]
            specific_audio_features = audio_features[:, audio_features.size(1) // 2:]
        
        if self.has_visual and visual_features is not None:
            # Split visual features into common and specific parts
            common_visual_features = visual_features[:, :visual_features.size(1) // 2]
            specific_visual_features = visual_features[:, visual_features.size(1) // 2:]

        # Generate modality-common features if z is provided (for knowledge distillation)
        generated_common_features = None
        generated_common_pred = None

        if z is not None:
            generated_common_features = self.generator(z)  # Generated audio common features
            generated_common_pred = self.common_classifier(generated_common_features)  # Generated audio common features

        # Process each modality's common features with their classifiers
        final_audio_pred = None
        final_visual_pred = None
        mean_common_features = 0
        modality = 0

        if common_audio_features is not None:
            modality += 1
            mean_common_features += common_audio_features if mean_common_features is not None else 0
            common_audio_pred = self.audio_common_classifier(common_audio_features)
            specific_audio_pred = self.audio_specific_classifier(specific_audio_features)
            final_audio_pred = common_audio_pred + specific_audio_pred  # Combining both predictions

        if common_visual_features is not None:
            modality += 1
            mean_common_features += common_visual_features if mean_common_features is not None else 0
            common_visual_pred = self.visual_common_classifier(common_visual_features)
            specific_visual_pred = self.visual_specific_classifier(specific_visual_features)
            final_visual_pred = common_visual_pred + specific_visual_pred  # Combining both predictions

        # Normalize mean_common_features by the number of contributing modalities
        if modality > 0:
            mean_common_features = mean_common_features / modality  

        # Compute final prediction by averaging predictions of all classifiers
        if final_audio_pred is not None and final_visual_pred is not None:
            final_prediction = (final_audio_pred + final_visual_pred) 
        elif final_audio_pred is not None:
            final_prediction = final_audio_pred
        elif final_visual_pred is not None:
            final_prediction = final_visual_pred
        else:
            final_prediction = None  # No valid predictions

        

        return final_prediction, (final_audio_pred, final_visual_pred, common_audio_features, common_visual_features, specific_audio_features, specific_visual_features, generated_common_features, generated_common_pred, mean_common_features)
    def compute_loss(self, audio_input, visual_input, labels, z, alpha1=1.0, alpha2=1.0, alpha_gen=1.0, beta=1.0, alpha_kd=1.0):
        # Forward pass
        final_prediction, (final_audio_pred, final_visual_pred, common_audio_features, common_visual_features, specific_audio_features, specific_visual_features, generated_common_features, generated_common_pred, mean_common_features) = self(audio_input, visual_input, z)

        # Initialize losses to zero
        similarity_loss = 0.0
        auxiliary_loss = 0.0
        difference_loss = 0.0
        generation_loss = 0.0
        kd_loss = 0.0  # Knowledge Distillation Loss

        # 1) Knowledge Distillation Loss (using the local generator)
        if common_audio_features is not None:
            kd_loss += self.compute_knowledge_distillation_loss(common_audio_features, z)
        if common_visual_features is not None:
            kd_loss += self.compute_knowledge_distillation_loss(common_visual_features, z)

        # 2) Similarity Loss (F_sim_k)
        if common_audio_features is not None and common_visual_features is not None:
            kl_loss_audio = self.compute_kl_divergence(common_audio_features, common_visual_features)
            similarity_loss = kl_loss_audio / 2  # Normalized by the number of modalities

        # 3) Auxiliary Classification Loss (F_cls_k)
        if common_audio_features is not None:
            auxiliary_loss += self.compute_auxiliary_classification_loss(common_audio_features, labels)
        if common_visual_features is not None:
            auxiliary_loss += self.compute_auxiliary_classification_loss(common_visual_features, labels)

        # 4) Difference Loss (F_dif_k) - Orthogonality between common and specific features
        if common_audio_features is not None and specific_audio_features is not None:
            difference_loss += self.compute_difference_loss(common_audio_features, specific_audio_features)
        if common_visual_features is not None and specific_visual_features is not None:
            difference_loss += self.compute_difference_loss(common_visual_features, specific_visual_features)

        # 5) Generation Loss (F_gen_k)
        if generated_common_features is not None:
            generation_loss += self.compute_generation_loss(generated_common_features, generated_common_pred, mean_common_features, labels, beta) 

        # 6) Total Loss (F_dec_k)
        total_loss = alpha1 * similarity_loss + alpha2 * difference_loss + auxiliary_loss + alpha_gen * generation_loss + alpha_kd * kd_loss
        return total_loss, similarity_loss, auxiliary_loss, difference_loss, generation_loss, kd_loss

    def compute_knowledge_distillation_loss(self, common_features, z):
        # Generate modality-common features using the local generator (input noise z)
        generated_features = self.generator(z)  # Generate modality-common features from noise
    
        # Pass both real common features and generated features through the common classifier
        common_features_pred = self.common_classifier(common_features)
        generated_features_pred = self.common_classifier(generated_features)
    
        # Apply softmax to both the predicted features
        softmax_common_features = F.softmax(common_features_pred, dim=-1)
        softmax_generated_features = F.softmax(generated_features_pred, dim=-1)
    
        # Compute KL divergence between the softmax outputs of the real and generated features
        kd_loss = F.kl_div(softmax_common_features.log(), softmax_generated_features, reduction='batchmean')
    
        return kd_loss

    def compute_kl_divergence(self, common_audio_features, common_visual_features):
        # Apply softmax to features and compute KL divergence
        softmax_audio = F.softmax(common_audio_features, dim=-1)
        softmax_visual = F.softmax(common_visual_features, dim=-1)
        kl_divergence = F.kl_div(softmax_audio.log(), softmax_visual, reduction='batchmean')
        return kl_divergence

    def compute_auxiliary_classification_loss(self, common_features, labels):
        # Cross-entropy loss for auxiliary classification
        return F.cross_entropy(self.common_classifier(common_features), labels)

    def compute_difference_loss(self, common_features, specific_features):
        # Orthogonality loss to ensure modality-common and modality-specific features are distinct
        return torch.norm(torch.matmul(common_features.T, specific_features), p='fro')**2

    def compute_generation_loss(self, generated_common_features, generated_common_pred, mean_common_features, labels, beta):
        # Mean squared error loss to ensure the generated features align with the true features
        generated_common_pred = F.softmax(generated_common_pred, dim=-1)
        return F.cross_entropy(generated_common_pred, labels) + beta * F.mse_loss(generated_common_features, mean_common_features)



