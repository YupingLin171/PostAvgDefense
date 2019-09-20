# -*- coding: utf-8 -*-

import torch
import robustml
import numpy as np
from foolbox.models import PyTorchModel
from robustml_portal import attacks as atk
from robustml_portal import postAveragedModels as pamdl

device = torch.device("cuda:0")

# setup test model
model = pamdl.pa_resnet110_config1()
model.to(device)
model.eval()

# setup attacker
victim_model = PyTorchModel(model.model, (0,1), 10, device=device, preprocessing=(np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))))
attack = atk.fgsmAttack(victim_model)

# setup data provider
test_data_path = './cifar10/cifar-10-batches-py/test_batch'
prov = robustml.provider.CIFAR10(test_data_path)

# evaluate performance
atk_success_rate = robustml.evaluate.evaluate(model, attack, prov, start=0, end=100)
print('Overall attack success rate: %.4f' % atk_success_rate)