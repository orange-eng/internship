import torch.nn as nn
import torch

VGG_select = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self,features,num_classes=1000,init_weights=False):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512*7*7,out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048,num_classes)
        )
        if init_weights:
            self.__initialize_weight()
    def forward(self,x):
        x = self.features(x)                #input[3,224,224]   output[512,7,7]
        x = torch.flatten(x,start_dim=1)    #output[512*7*7]
        x = self.classifier(x)              #output[num_classes]
        return x


    def __initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)   


def make_features(_VGG_select: list):
    layers = []
    input_channels = 3
    for v in _VGG_select:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(input_channels, v, kernel_size=3,padding=1)
            layers += [conv2d,nn.ReLU(True)]
            input_channels = v           
    return nn.Sequential(*layers)


def vgg(model_name='vgg16',**kwargs):
    try:
        _VGG_select=VGG_select[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(_VGG_select),**kwargs)
    return model

'''
附录
*args和**kwargs一般是用在函数定义的时候。
二者的意义是允许定义的函数接受任意数目的参数。
# 也就是说我们在函数被调用前并不知道也不限制将来函数可以接收的参数数量。
在这种情况下我们可以使用*args和**kwargs。
'''
inputs = torch.rand([32,3,224,224])
net = vgg()
print(net)
outputs = net(inputs)
print(outputs)

    

