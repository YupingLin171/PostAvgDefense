# -*- coding: utf-8 -*-

import time
import os
import os.path
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.cuda as cuda
import torchvision.transforms as transforms
import torchvision.utils as utl
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets

from foolbox.models import PyTorchModel
import foolbox.criteria as crt
import foolbox.attacks as attacks
import foolbox.distances as distances
import foolbox.adversarial as adversarial

import KNDefense as kndef
import visualHelper as vh
import resnet as ylrsn

# ==============================
# set parameters
device = torch.device("cuda:0")
cudnn.benchmark = True
use_targeted_attack = False
use_top5_criterion = False  # <<<<<<<<<<<<<<<<<<<<<
use_early_stop = False
include_original_img = True
fix_test_set = False
rnd_seed = 20190501

# number of adversarial samples to test
kImgs = 'all'
K = 15
R = 6
r = [[R/3, 2*R/3, R]]
attack_range = 8 / 255 # 0.1
batchSize = 1000
nClasses = 10
num_targets_retry = 3  # used only in targeted attack
range_type = 'Linf' # options = ['L2', 'Linf']
attack_method = 'FGSM' # options = ['LBFGS', 'FGSM', 'PGD', 'DeepFool', 'CW']
sample_method = ['random'] # options = ['random', 'approx_cifar10']
shift_direction = 'both' # options = ['both', 'inc', 'dec']
vote_method = 'avg_softmax' # options = ['avg_feat', 'most_vote', 'weighted_feat', 'avg_softmax']

# define paths
dataDir = '/local/ssd/yuping/videoLearning/cifar10'
trainedModelPath = './trainedModel/resnet110.th'
showCharts = True
showNoiseMaps = False
showCorrectSamples = False
chartSaveDir = './results_CIFAR10_FGSM_N10000/expm_random_sfmx_K15_r2-4-6_linf8_255'

# normalization preprocessing required by the pretrained model
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

# =============================
# load dataset
# NOT performing normalization here because the attacking step needs the original image
dataset = datasets.CIFAR10(dataDir, train=False, transform=transforms.ToTensor())
if kImgs == 'all':
    kImgs = len(dataset)

# setup pretrained model
checkpoint = torch.load(trainedModelPath)
paramDict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    # remove 'module.' prefix introduced by DataParallel, if any
    if k.startswith('module.'):
        paramDict[k[7:]] = v
model = ylrsn.resnet110()
model.load_state_dict(paramDict)
model = model.to(device)
model = model.eval()

# define victim model 
victim_model = PyTorchModel(model, (0,1), nClasses, device=device, preprocessing=(np.array(mean).reshape((3, 1, 1)), np.array(std).reshape((3, 1, 1))))
if use_targeted_attack:
    # this setup is only for compatibility, the actually setup is done in each iteration
    target_class = 0
    adv_criterion = crt.TargetClass(target_class)
elif use_top5_criterion:
    adv_criterion = crt.TopKMisclassification(5)
else:
    adv_criterion = crt.Misclassification()

# set up attacker
if attack_method == 'LBFGS':
    # seems doesn't support early stopping for this attack
    adv_attacker = attacks.LBFGSAttack(victim_model, adv_criterion)
elif attack_method == 'FGSM':
    if use_early_stop:
        adv_attacker = attacks.GradientSignAttack(victim_model, adv_criterion, distance=distances.Linfinity, threshold=attack_range)
    else:
        adv_attacker = attacks.GradientSignAttack(victim_model, adv_criterion)
elif attack_method == 'PGD':
    if use_early_stop:
        adv_attacker = attacks.RandomStartProjectedGradientDescentAttack(victim_model, adv_criterion, distance=distances.Linfinity, threshold=attack_range)
    else:
        adv_attacker = attacks.RandomStartProjectedGradientDescentAttack(victim_model, adv_criterion, distance=distances.Linfinity)
elif attack_method == 'DeepFool':
    if use_early_stop:
        adv_attacker = attacks.DeepFoolAttack(victim_model, adv_criterion, distance=distances.Linfinity, threshold=attack_range)
    else:
        adv_attacker = attacks.DeepFoolAttack(victim_model, adv_criterion)
elif attack_method == 'CW':
    if use_early_stop:
        adv_attacker = attacks.CarliniWagnerL2Attack(victim_model, adv_criterion, distance=distances.Linfinity, threshold=attack_range)
    else:
        adv_attacker = attacks.CarliniWagnerL2Attack(victim_model, adv_criterion)

# =============================
# iterate over input samples
n_advSp = 0
n_advSp_dfs = 0
n_missed_oriSet = 0
n_missed_attSet = 0
n_missed_oriSet_dfs = 0
n_missed_attSet_dfs = 0
n_top5missed_oriSet = 0
n_top5missed_attSet = 0
n_top5missed_oriSet_dfs = 0
n_top5missed_attSet_dfs = 0
n_examed = 0
perts = []
if fix_test_set:
    np.random.seed(rnd_seed)
inx_sq = np.random.permutation(len(dataset))
for inx in inx_sq:
    # exit when examinated enough number of adversarial samples
    if n_examed >= kImgs:
        break
    n_examed = n_examed + 1
    
    t_0 = time.time()
    cuda.empty_cache()  # clear caches
    
    # get input sample
    sp, lb = dataset[inx]
    
    # ----------------------------------------------------
    # predict the original sample with the original model
    t_1 = time.time()
    top5Scores = np.argsort(victim_model.predictions(sp.numpy()))[-5:]
    pred = top5Scores[-1]
    t_oriPred = time.time() - t_1
    
    cls_correct_ori = True
    if use_top5_criterion:
        if pred != lb:
            n_missed_oriSet = n_missed_oriSet + 1
        if lb not in top5Scores:
            cls_correct_ori = False
            n_top5missed_oriSet = n_top5missed_oriSet + 1
    else:
        if pred != lb:
            cls_correct_ori = False
            n_missed_oriSet = n_missed_oriSet + 1
        if lb not in top5Scores:
            n_top5missed_oriSet = n_top5missed_oriSet + 1
        
    # ----------------------------------------------------
    # predict the original sample with the defense mechanism
    t_2 = time.time()
    x = normalize(sp).unsqueeze(0)
    x_squad = []
    for i in range(len(r)):
        x_squad.append(kndef.formSquad_resnet(sample_method[i], model, x, K, r[i], direction=shift_direction, device=device, includeOriginal=include_original_img))
    
    y = [None] * len(r)
    y_feats = [None] * len(r)
    entropyScores = torch.ones(len(r)) * torch.tensor(float('Inf'))
    for i in range(len(r)):
        if sample_method[i].startswith('feats'):
            y[i], y_feats[i] = kndef.integratedForward_cls(model, x_squad[i], batchSize, nClasses, device=device, voteMethod=vote_method)
        else:
            y[i], y_feats[i] = kndef.integratedForward(model, x_squad[i], batchSize, nClasses, device=device, voteMethod=vote_method)
        entropyScores[i] = kndef.checkEntropy(y[i])
    
    selectedInx = torch.argmin(entropyScores).item()
    y = y[selectedInx]
    y_feats = y_feats[selectedInx]
    
    top5Scores_dfs = torch.argsort(y.squeeze())[-5:]
    pred_dfs = top5Scores_dfs[-1]
    t_defPred = time.time() - t_2
    
    cls_correct_dfs = True
    if use_top5_criterion:
        if pred_dfs != lb:
            n_missed_oriSet_dfs = n_missed_oriSet_dfs + 1
        if lb not in top5Scores_dfs:
            cls_correct_dfs = False
            n_top5missed_oriSet_dfs = n_top5missed_oriSet_dfs + 1
    else:
        if pred_dfs != lb:
            cls_correct_dfs = False
            n_missed_oriSet_dfs = n_missed_oriSet_dfs + 1
        if lb not in top5Scores_dfs:
            n_top5missed_oriSet_dfs = n_top5missed_oriSet_dfs + 1
    
    # ----------------------------------------------------
    # generate adversarial example for the correctly classified samples
    generated_adv = False
    if cls_correct_ori:
        # generate adversarial sample
        if use_targeted_attack:
            # randomly select target classes
            target_classes = random.sample(range(nClasses), k=num_targets_retry + 1)
            target_classes = [t for t in target_classes if t != lb]
            
            n_tried = 0
            while n_tried < num_targets_retry and not generated_adv:
                adv_criterion = crt.TargetClass(target_classes[n_tried])
                if range_type == 'Linf':
                    adv_obj = adversarial.Adversarial(victim_model, adv_criterion, sp.numpy(), lb, distance=distances.Linfinity)
                else:
                    adv_obj = adversarial.Adversarial(victim_model, adv_criterion, sp.numpy(), lb, distance=distances.MeanSquaredDistance)
                
                adv_sp = adv_attacker(adv_obj)
                
                if adv_sp is not None:
                    nse_sp = torch.abs(sp - torch.from_numpy(adv_sp))
                    L2_pert = torch.norm(nse_sp).item()
                    if range_type == 'Linf':
                        d_pert = torch.max(nse_sp).item()
                    else:
                        d_pert = L2_pert
            
                    if d_pert <= attack_range:
                        generated_adv = True
                
                n_tried = n_tried + 1
        else:
            adv_sp = adv_attacker(sp.numpy(), lb)
        
            if adv_sp is not None:
                nse_sp = torch.abs(sp - torch.from_numpy(adv_sp))
                L2_pert = torch.norm(nse_sp).item()
                if range_type == 'Linf':
                    d_pert = torch.max(nse_sp).item()
                else:
                    d_pert = L2_pert
            
                if d_pert <= attack_range:
                    generated_adv = True
    
    # ----------------------------------------------------
    # predict the adversarial sample with the original model
    if generated_adv:
        adv_scores = victim_model.predictions(adv_sp)
        top5Scores_adv = np.argsort(adv_scores)[-5:]
        adv_pred = top5Scores_adv[-1]
    
        if adv_criterion.is_adversarial(adv_scores, lb):
            n_advSp = n_advSp + 1
        else:
            # skip when the found sample is not effectively an adversarial sample
            generated_adv = False
    
    # ----------------------------------------------------
    # predict the adversarial sample with the defense mechanism
    if generated_adv:
        t_3 = time.time()
        adv_sp = torch.from_numpy(adv_sp)
        adv_x = normalize(adv_sp).unsqueeze(0)
    
        adv_x_squad = []
        for i in range(len(r)):
            adv_x_squad.append(kndef.formSquad_resnet(sample_method[i], model, adv_x, K, r[i], direction=shift_direction, device=device, includeOriginal=include_original_img))
        
        adv_y = [None] * len(r)
        adv_y_feats = [None] * len(r)
        adv_entropyScores = torch.ones(len(r)) * torch.tensor(float('Inf'))
        for i in range(len(r)):
            if sample_method[i].startswith('feats'):
                adv_y[i], adv_y_feats[i] = kndef.integratedForward_cls(model, adv_x_squad[i], batchSize, nClasses, device=device, voteMethod=vote_method)
            else:
                adv_y[i], adv_y_feats[i] = kndef.integratedForward(model, adv_x_squad[i], batchSize, nClasses, device=device, voteMethod=vote_method)
            adv_entropyScores[i] = kndef.checkEntropy(adv_y[i])
        
        selectedInx = torch.argmin(adv_entropyScores).item()
        adv_y = adv_y[selectedInx]
        adv_y_feats = adv_y_feats[selectedInx]
        
        top5Scores_adv_dfs = torch.argsort(adv_y.squeeze())[-5:]
        adv_pred_dfs = top5Scores_adv_dfs[-1]
        t_dfs = time.time() - t_3

        if use_targeted_attack and adv_pred_dfs == lb:
            n_advSp_dfs = n_advSp_dfs + 1
            perts.append((L2_pert, 1))
        elif not use_targeted_attack and not adv_criterion.is_adversarial(adv_y.squeeze().numpy(), lb):
            n_advSp_dfs = n_advSp_dfs + 1
            perts.append((L2_pert, 1))
        else:
            perts.append((L2_pert, 0))
        
        if adv_pred != lb:
            n_missed_attSet = n_missed_attSet + 1
        if lb not in top5Scores_adv:
            n_top5missed_attSet = n_top5missed_attSet + 1
        if adv_pred_dfs != lb:
            n_missed_attSet_dfs = n_missed_attSet_dfs + 1
        if lb not in top5Scores_adv_dfs:
            n_top5missed_attSet_dfs = n_top5missed_attSet_dfs + 1
    else:
        if pred != lb:
            n_missed_attSet = n_missed_attSet + 1
        if lb not in top5Scores:
            n_top5missed_attSet = n_top5missed_attSet + 1
        if pred_dfs != lb:
            n_missed_attSet_dfs = n_missed_attSet_dfs + 1
        if lb not in top5Scores_dfs:
            n_top5missed_attSet_dfs = n_top5missed_attSet_dfs + 1
    
    # ----------------------------------------------------
    # show prediction statistics with charts
    if showCharts:
        if not cls_correct_dfs:
            if cls_correct_ori:
                vh.plotPredStats(y_feats, lb, image=sp.numpy().transpose(1, 2, 0), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}_missed.png"))
            else:
                vh.plotPredStats(y_feats, lb, image=sp.numpy().transpose(1, 2, 0), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}_bothMissed.png"))
        elif showCorrectSamples:
            vh.plotPredStats(y_feats, lb, image=sp.numpy().transpose(1, 2, 0), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}.png"))
        if generated_adv:
            if (not use_top5_criterion and adv_pred_dfs != lb) or (use_top5_criterion and lb not in top5Scores_adv_dfs):
                if showNoiseMaps:
                    vh.plotPredStats(adv_y_feats, lb, image=adv_sp.numpy().transpose(1, 2, 0), noiseImage=np.clip(nse_sp.numpy().transpose(1, 2, 0) * 150, 0, 1), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}-adv_missed.png"))
                else:
                    vh.plotPredStats(adv_y_feats, lb, image=adv_sp.numpy().transpose(1, 2, 0), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}-adv_missed.png"))
            elif showCorrectSamples:
                if showNoiseMaps:
                    vh.plotPredStats(adv_y_feats, lb, image=adv_sp.numpy().transpose(1, 2, 0), noiseImage=np.clip(nse_sp.numpy().transpose(1, 2, 0) * 150, 0, 1), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}-adv.png"))
                else:
                    vh.plotPredStats(adv_y_feats, lb, image=adv_sp.numpy().transpose(1, 2, 0), savePath=os.path.join(chartSaveDir, f"stats_img{inx:05d}-adv.png"))
    
    # ----------------------------------------------------
    # show statistics
    if generated_adv:
        print('============Image %05d===============' % inx)
        print('[img%05d] label: %d;\toriginal prediction: %d, time:%.2fsec;\toriginal prediction with defence: %d, time:%.2fsec;\tadversarial prediction: %d;\tadversarial prediction with defence: %d;\tperturbation distance: %.2E;\ttotal time used: %.2fsec;\ttime used to defense: %.2fsec;' % (inx, lb, pred, t_oriPred, pred_dfs, t_defPred, adv_pred, adv_pred_dfs, L2_pert, time.time() - t_0, t_dfs))
    else:
        print('============Image %05d===============' % inx)
        print('[img%05d] label: %d;\toriginal prediction: %d;\toriginal prediction with defence: %d;\ttotal time used: %.2fsec;' % (inx, lb, pred, pred_dfs, time.time() - t_0))

# ----------------------------------------------------
# show overall statistics
ptbs = np.asarray(perts)[:, 0]
print('=====================================')
print('%d images are examinated;\n%d(%.4f) adversarial samples are successfully generated;\n%.4f(%d) of the adversarial samples are successfully defensed;\nmin perturbation: %.2e;\tavg perturbation: %.2e;\tmax perturbation: %.2e\nClassification accuracy on the original dataset:\noriginal model: top1: %.4f(%d);\ttop5: %.4f(%d);\nmodel with defense: top1: %.4f(%d);\ttop5: %.4f(%d);\nClassification accuracy on the adversarial attacked dataset:\noriginal model: top1: %.4f(%d);\ttop5: %.4f(%d);\nmodel with defense: top1: %.4f(%d);\ttop5: %.4f(%d);' % (n_examed, n_advSp, n_advSp / n_examed, n_advSp_dfs / n_advSp, n_advSp_dfs, np.min(ptbs), np.mean(ptbs), np.max(ptbs), (n_examed - n_missed_oriSet) / n_examed, n_examed - n_missed_oriSet, (n_examed - n_top5missed_oriSet) / n_examed, n_examed - n_top5missed_oriSet, (n_examed - n_missed_oriSet_dfs) / n_examed, n_examed - n_missed_oriSet_dfs, (n_examed - n_top5missed_oriSet_dfs) / n_examed, n_examed - n_top5missed_oriSet_dfs, (n_examed - n_missed_attSet) / n_examed, n_examed - n_missed_attSet, (n_examed - n_top5missed_attSet) / n_examed, n_examed - n_top5missed_attSet, (n_examed - n_missed_attSet_dfs) / n_examed, n_examed - n_missed_attSet_dfs, (n_examed - n_top5missed_attSet_dfs) / n_examed, n_examed - n_top5missed_attSet_dfs))

# plot perturbation distribution
vh.plotPerturbationDistribution(perts, savePath=os.path.join(chartSaveDir, f"dist_perts.png"))
