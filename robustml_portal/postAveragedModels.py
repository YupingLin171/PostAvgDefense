# -*- coding: utf-8 -*-

import robustml
import numpy as np
from collections import OrderedDict
import PADefense as padef
import resnetSmall as rnsmall

import torch
import torchvision.models as mdl
import torchvision.transforms as transforms

class PostAveragedResNet152(robustml.model.Model):
    def __init__(self, K, R, eps):
        self._model = mdl.resnet152(pretrained=True)
        self._dataset = robustml.dataset.ImageNet((224, 224, 3))
        self._threat_model = robustml.threat_model.Linf(epsilon=eps)
        self._K = K
        self._r = [R/3, 2*R/3, R]
        self._sample_method = 'random'
        self._vote_method = 'avg_softmax'
    
    @property
    def model(self):
        return self._model
    
    @property
    def dataset(self):
        return self._dataset
        
    @property
    def threat_model(self):
        return self._threat_model
        
    def classify(self, x):
        # transpose x to accommodate pytorch's axis arrangement convention
        x = torch.as_tensor(x).transpose(0, 2)
        
        # preprocess data
        x = self._preprocess(x).unsqueeze(0)
        
        # gather neighbor samples
        x_squad = padef.formSquad_resnet(self._sample_method, self._model, x, self._K, self._r)
        
        # forward with a batch of neighbors
        feat, _ = padef.integratedForward(self._model, x_squad, batchSize=100, nClasses=1000, voteMethod=self._vote_method)
        
        # get predicted class
        prediction = torch.argmax(feat.squeeze()).item()
        
        return prediction
        
    def _preprocess(self, image):
        # normalization used by pre-trained model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return normalize(image)
        
    def to(self, device):
        self._model = self._model.to(device)
        
    def eval(self):
        self._model = self._model.eval()
        
        
def pa_resnet152_config1():
    return PostAveragedResNet152(K=15, R=30, eps=8/255)

    
class PostAveragedResNet110(robustml.model.Model):
    def __init__(self, K, R, eps):
        # load model state dict
        checkpoint = torch.load('./trainedModel/resnet110.th')
        paramDict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            # remove 'module.' prefix introduced by DataParallel, if any
            if k.startswith('module.'):
                paramDict[k[7:]] = v
        self._model = rnsmall.resnet110()
        self._model.load_state_dict(paramDict)
        
        self._dataset = robustml.dataset.CIFAR10()
        self._threat_model = robustml.threat_model.Linf(epsilon=eps)
        self._K = K
        self._r = [R/3, 2*R/3, R]
        self._sample_method = 'random'
        self._vote_method = 'avg_softmax'
    
    @property
    def model(self):
        return self._model
    
    @property
    def dataset(self):
        return self._dataset
        
    @property
    def threat_model(self):
        return self._threat_model
        
    def classify(self, x):
        # transpose x to accommodate pytorch's axis arrangement convention
        x = torch.as_tensor(np.transpose(x, (2, 0, 1)))
        
        # preprocess data
        x = self._preprocess(x).unsqueeze(0)
        
        # gather neighbor samples
        x_squad = padef.formSquad_resnet(self._sample_method, self._model, x, self._K, self._r)
        
        # forward with a batch of neighbors
        feat, _ = padef.integratedForward(self._model, x_squad, batchSize=1000, nClasses=10, voteMethod=self._vote_method)
        
        # get predicted class
        prediction = torch.argmax(feat.squeeze()).item()
        
        return prediction
        
    def _preprocess(self, image):
        # normalization used by pre-trained model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return normalize(image)
        
    def to(self, device):
        self._model = self._model.to(device)
        
    def eval(self):
        self._model = self._model.eval()
        
        
def pa_resnet110_config1():
    return PostAveragedResNet110(K=15, R=6, eps=8/255)
