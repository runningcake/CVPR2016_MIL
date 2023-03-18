import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

class DIYNet(nn.Module): # resnet50
    def __init__(self, backbone, classifier, num_classes=2):
        super(DIYNet, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        features = self.backbone(x)
        x = self.classifier(features)
        return x

NUM_CLASSES = 3 # ADC, SCC, ADC_mix
backbone = models.resnet50(pretrained=True)
return_layers = {'layer4': 'out'}
backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(2048, NUM_CLASSES, 1) # resnet50 layer4's output has 2048 channels
)
model_resnet50_as_backbone = DIYNet(backbone, classifier)