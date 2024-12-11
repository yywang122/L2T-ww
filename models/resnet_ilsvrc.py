import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','AlexNet', 'alexnet',
           'resnet152','VG16', 'vg11', 'vg11_bn', 'vg13', 'vg13_bn', 'VG16','vg16', 'vg16_bn',
    'vg19_bn', 'vg19', 'vg9_bn']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(AlexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.classifier = nn.Sequential(
        	nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
    	
    	f1 = self.conv1(x) #....features.0.weight & features.0.bias
    	
    	r1 = self.relu(f1)
    	p1 = self.maxpool(r1)
    	f2 = self.conv2(p1) #....features.3.weight & features.3.bias
    	r2 = self.relu(f2)
    	p2 = self.maxpool(r2)
    	f3 = self.conv3(p2)
    	r3 = self.relu(f3)
    	f4 = self.conv4(r3)
    	r4 = self.relu(f4)
    	f5 = self.conv5(r4)
    	r5 = self.relu(f5)
    	p5 = self.maxpool(r5)
    	f7 = p5.view(p5.size(0), -1)
    	f8 = self.classifier(f7)
    	return f7,[p1,p2,r3,r4,p5]   
    def forward_with_features(self, x):
        return self.forward(x)
'''     
        for k,v in model_zoo.load_url(model_urls['alexnet']).items():
        	print({k})   # features.0.weight , features.0.bias .....
        	p=k.replace('features.0','conv1').replace('features.0','conv1').replace('features.3','conv2').replace('features.6','conv3').replace('features.8','conv4').replace('features.10','conv5')
        	print(p)  # conv1.weight , conv1.bias .....
        '''    
    
def alexnet(pretrained=False, **kwargs):

    model = AlexNet(**kwargs)
    if pretrained:
    
        #model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        
        model.load_state_dict({k.replace('features.0','conv1').replace('features.0','conv1').replace('features.3','conv2').replace('features.6','conv3').replace('features.8','conv4').replace('features.10','conv5'):v for k,v in model_zoo.load_url(model_urls['alexnet']).items()})
        	

       
        	
    return model




class v9(nn.Module):
    def __init__(self, num_classes=1000):
        super(v9, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
    	
    	f1 = self.conv1(x)
    	r1 = self.relu(f1)
    	p1 = self.maxpool(r1)
    	
    	f2 = self.conv2(p1)
    	r2 = self.relu(f2)
    	p2 = self.maxpool(r2)
    	
    	f3 = self.conv3(p2)
    	r3 = self.relu(f3)
    	
    	f4 = self.conv4(r3)
    	r4 = self.relu(f4)
    	p4 = self.maxpool(r4)
    	
    	f5 = self.conv5(p4)
    	r5 = self.relu(f5)
    	
    	f6 = self.conv2(r5)
    	r6 = self.relu(f6)
    	p6 = self.maxpool(r6)
    	
    	f7 = self.conv3(p6)
    	r7 = self.relu(f7)
    	
    	f8 = self.conv4(r7)
    	r8 = self.relu(f8)
    	p8 = self.maxpool(r8)
    	
    	f9 = p8.view(p8.size(0), -1)
    	f10 = self.classifier(f9)
    	
    	
    	return f10,[f1,f2,f3,f5,f8]   
    def forward_with_features(self, x):
        return self.forward(x)
    
    
def vgg9(pretrained=False, **kwargs):

    model = v9(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['v9']))
        
    return model












'''
class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):#imagenet数量
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
         #需要针对上一层改变view
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        
        self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        f6 = f5.view(f5.size(0), -1)
        f7 = self.layer6(f6)
        f8 = self.layer7(f7)
        f9 = self.layer8(f8)
        return f9, [f1, f2, f4, f5]
        
        
    def forward_with_features(self, x):
        return self.forward(x)
'''



class VG16(nn.Module):
    def __init__(self, num_classes = 1000):#imagenet数量
    	super(VG16,self).__init__()
    	
    	self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1, padding=1)
    	self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
    	self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
    
    	self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
    	self.conv4 = nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
    	self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
    
    	self.conv5 = nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
    	self.conv6 = nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
    	self.conv7 = nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
    	self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
    	
    	self.conv8 = nn.Conv2d(256,512,kernel_size=3,stride=1, padding=1)
    	self.conv9 = nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
    	self.conv10 = nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
    	self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)
    	
    	self.conv11 = nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
    	self.conv12 = nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
    	self.conv13 = nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
    	self.maxpool5 = nn.MaxPool2d(kernel_size=2,stride=2)
    	self.fc1 = nn.Linear(512*7*7,4096)
    	self.fc2 = nn.Linear(4096,4096)
    	self.fc3 = nn.Linear(4096,num_classes)
    	self.relu= nn.ReLU(inplace=True)
    def forward(self,x):
    	f1 = self.conv1(x)
    	r1 = self.relu(f1)
    	f2 = self.conv2(r1)
    	r2 = self.relu(f2)
    	p1 = self.maxpool(r2)
    	
    	f3 = self.conv3(p1)
    	r3 = self.relu(f3)
    	f4 = self.conv4(r3)
    	r4 = self.relu(f4)
    	p2 = self.maxpool(r4)
    	
    	f5 = self.conv5(p2)
    	r5 = self.relu(f5)
    	f6 = self.conv6(r5)
    	r6 = self.relu(f6)
    	f7 = self.conv7(r6)
    	r7 = self.relu(f7)
    	p3 = self.maxpool(r7)
    	
    	f8 = self.conv8(p3)
    	r8 = self.relu(f8)
    	f9 = self.conv9(r8)
    	r9 = self.relu(f9)
    	f10 = self.conv10(r9)
    	r10 = self.relu(f10)
    	p4 = self.maxpool(r10)
    	
    	f11 = self.conv11(p4)
    	r11 = self.relu(f11)
    	f12 = self.conv12(r11)
    	r12 = self.relu(f12)
    	f13 = self.conv13(r12)
    	r13 = self.relu(f13)
    	p5 = self.maxpool(r13)
    	f14 = p5.view(p5.size(0), -1)
    	
    	f15 = self.fc1(f14)
    	r14 = self.relu(f15)
    	f16= self.fc2(r14)
    	r15= self.relu(f16)
    	f17= self.fc3(r15)
    	
    	return f17,[f2,f4,f7,f10,f13]
    def forward_with_features(self, x):
        return self.forward(x)
        
'''
        model.load_state_dict({k.replace('features.0','conv1').replace('features.2','conv2').replace('features.5','conv3').replace('features.7','conv4').replace('features.10','conv5').replace('features.12','conv6').replace('features.14','conv7').replace('features.17','conv8').replace('features.19','conv9').replace('features.21','conv10').replace('features.24','conv11').replace('features.26','conv12').replace('features.28','conv13'):v for k,v in model_zoo.load_url(model_urls['vg16']).items()})
        '''
def vg16(pretrained=False, **kwargs):

    model = VG16(**kwargs)
    if pretrained:
    	#model.load_state_dict(model_zoo.load_url(model_urls['vg16']))
    	model.load_state_dict({k.replace('features.0','conv1').replace('features.2','conv2').replace('features.5','conv3').replace('features.7','conv4').replace('features.10','conv5').replace('features.12','conv6').replace('features.14','conv7').replace('features.17','conv8').replace('features.19','conv9').replace('21','10').replace('24','11').replace('26','12').replace('28','13').replace('classifier.0','fc1').replace('classifier.3','fc2').replace('classifier.6','fc3'):v for k,v in model_zoo.load_url(model_urls['vg16']).items()})
        
        
    return model

	

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, meta=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lwf = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.conv1(x)
        b1 = self.bn1(f1)
        r1 = self.relu(b1)
        p1 = self.maxpool(r1)

        f2 = self.layer1(p1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)

        f6 = self.avgpool(f5)
        f6 = f6.view(f6.size(0), -1)
        f7 = self.fc(f6)

        return f7, [r1, f2, f3, f4, f5]

    def forward_with_features(self, x):
        return self.forward(x)


def resnet18(pretrained=False, meta=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if meta:
        model = ResNet_meta(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, meta=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if meta:
        model = ResNet_meta(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
