
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
from torch import autograd
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets

from utils import *
import clip
from model import CustomCLIP 


from my_dataset import build_dataset
from my_dataset.utils import build_data_loader

from ood_utils.ood_tool import get_measures

    



def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/my_config.yaml", help='settings in yaml format')
    parser.add_argument('--is_train', type=int, default=1, help='1->train 0->test' )
    args = parser.parse_args()

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ["true", "1", "yes", "y"]
    return bool(v)


def build_negative_targets(target, num_classes):
    """Build multi-label targets for class-specific negative prompts.

    For an image with label y, the prompt "not y" should be low (0),
    while prompts "not other classes" should be high (1).
    """
    neg_targets = torch.ones(target.shape[0], num_classes, device=target.device)
    neg_targets.scatter_(1, target.view(-1, 1), 0.0)
    return neg_targets


def get_negative_pred_score(just_forced_logits, logits_neg, temperature=1.0):
    """Return negative-prompt confidence of the currently predicted positive class.

    just_forced_logits: [B, C], positive forced-prompt logits after normalization
    logits_neg: [B, C], negative-prompt logits after normalization
    """
    pred_cls = just_forced_logits.argmax(dim=1)
    neg_prob = torch.sigmoid(logits_neg / temperature)
    neg_pred_score = neg_prob.gather(1, pred_cls.view(-1, 1)).squeeze(1)
    return neg_pred_score


def compute_class_adaptive_margins(model, data_loader, num_classes, cfg, device):
    """Compute one margin for each ID class from frozen CLIP image prototypes.

    This is intentionally lightweight and training-only:
    1) extract global CLIP image features for few-shot training images;
    2) average features within each class to obtain class prototypes;
    3) measure each class's nearest-prototype similarity;
    4) shrink margin for visually confusing classes and enlarge it for separable classes.

    No extra hyperparameters such as m_min/m_max are introduced. Margins are centered
    around cfg['neg_margin'] and conservatively clipped to +/-20%.
    """
    base_margin = float(cfg.get('neg_margin', 0.1))
    if num_classes <= 1:
        return torch.full((num_classes,), base_margin, device=device)

    model_was_training = model.training
    model.eval()

    module = model.module if hasattr(model, 'module') else model
    image_encoder = module.image_encoder
    dtype = getattr(module, 'dtype', torch.float32)

    feature_dim = int(cfg['embed_dim'])
    feat_sum = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    print('Computing class-adaptive margins from frozen CLIP image prototypes...')
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            image_out = image_encoder(images.type(dtype))
            if isinstance(image_out, (tuple, list)):
                image_features = image_out[0]
            else:
                image_features = image_out

            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            feat_sum.index_add_(0, labels, image_features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))

    # Avoid division by zero. In normal few-shot training every class has samples.
    prototypes = feat_sum / counts.clamp_min(1.0).view(-1, 1)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sim_matrix = prototypes @ prototypes.t()
    sim_matrix.fill_diagonal_(-1e4)
    nearest_sim = sim_matrix.max(dim=1).values

    # If all classes have almost identical nearest similarities, keep the original margin.
    sim_std = nearest_sim.std(unbiased=False)
    if float(sim_std.item()) < 1e-6:
        margins = torch.full((num_classes,), base_margin, device=device)
    else:
        # High nearest_sim => this class is easily confused with another class => smaller margin.
        # Low nearest_sim  => this class is more separable => larger margin.
        z = (nearest_sim - nearest_sim.mean()) / (sim_std + 1e-6)
        factors = torch.clamp(1.0 - 0.2 * z, min=0.8, max=1.2)
        margins = base_margin * factors

    classnames = cfg.get('classnames', [str(i) for i in range(num_classes)])
    print('Class-adaptive margins around base neg_margin = {:.4f}'.format(base_margin))
    for idx, name in enumerate(classnames):
        print('  class {:02d} {:>28s} | nearest_sim={:.4f} | margin={:.4f}'.format(
            idx, str(name), float(nearest_sim[idx].item()), float(margins[idx].item())
        ))

    if model_was_training:
        model.train()

    return margins.detach()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    print(f"GPU num：{num_gpus}")

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['device'] = device
    cfg['is_train'] = args.is_train

    # ----- FA-Neg configs -----
    # Default values keep the original FA backward-compatible.
    cfg['use_negative_prompt'] = str2bool(cfg.get('use_negative_prompt', False))
    cfg['negative_template'] = cfg.get('negative_template', 'a photo of not a')
    cfg['negative_template_name'] = cfg.get('negative_template_name', 'notphoto')
    cfg['lambda_neg'] = float(cfg.get('lambda_neg', 0.0))
    cfg['lambda_margin'] = float(cfg.get('lambda_margin', 0.0))
    cfg['neg_margin'] = float(cfg.get('neg_margin', 0.1))
    cfg['alpha_neg'] = float(cfg.get('alpha_neg', 0.5))
    # Positive logits in the original code are divided by 100.0 at inference.
    # We use the same scale for negative logits in BCE/margin and test-time sigmoid.
    cfg['neg_loss_scale'] = float(cfg.get('neg_loss_scale', 100.0))

    # ----- Self-calibrated / correct-only margin configs -----
    # This is deliberately conservative: it does NOT change the test-time score.
    # It only gates the auxiliary negative losses during training.
    cfg['use_conf_gate'] = str2bool(cfg.get('use_conf_gate', False))
    cfg['conf_gate_gamma'] = float(cfg.get('conf_gate_gamma', 1.0))
    cfg['conf_gate_min'] = float(cfg.get('conf_gate_min', 0.5))
    cfg['margin_correct_only'] = str2bool(cfg.get('margin_correct_only', False))

    # ----- Class-adaptive margin config -----
    # It only changes the training-time margin value m_y.
    # Test-time MCM / GL-MCM scoring remains unchanged.
    cfg['use_class_adaptive_margin'] = str2bool(cfg.get('use_class_adaptive_margin', False))

    # CLIP 
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model = torch.nn.DataParallel(clip_model).to(device)
    clip_model = clip_model.module

    model_file_name_dict = {
        "RN50": "RN50",
        "RN101": "RN101",
        "RN50x4": "RN50x4",
        "RN50x16": "RN50x16",
        "ViT-B/32": "ViT-B-32",
        "ViT-B/16": "ViT-B-16",
    }

    model_path = os.path.join(cfg['clip_model_path'], model_file_name_dict[cfg['backbone']]+".pt" )
    model_temp = torch.jit.load(model_path, map_location=device ).eval()
    state_dict = model_temp.state_dict()


    cfg['embed_dim'] = state_dict["text_projection"].shape[1]
    # print(cfg['embed_dim'])

       

    K_name = str(cfg['K']).replace('.','-')
    lr_name = str(cfg['lr']).replace('.','')

    if cfg['use_negative_prompt']:
        lambda_neg_name = str(cfg['lambda_neg']).replace('.', '')
        lambda_margin_name = str(cfg['lambda_margin']).replace('.', '')
        model_name = 'FA_Neg_' + str(cfg['negative_template_name']) + '_lneg' + lambda_neg_name + '_lmar' + lambda_margin_name
        if cfg.get('use_conf_gate', False):
            gamma_name = str(cfg.get('conf_gate_gamma', 1.0)).replace('.', '')
            min_name = str(cfg.get('conf_gate_min', 0.5)).replace('.', '')
            model_name += '_cg' + gamma_name + '_min' + min_name
        if cfg.get('margin_correct_only', False):
            model_name += '_corronly'
        if cfg.get('use_class_adaptive_margin', False):
            model_name += '_cam'
    else:
        model_name = 'FA'

    cache_dir = os.path.join('./my_caches', cfg['id_dataset'], model_file_name_dict[cfg['backbone']], str(cfg['shots'])+'shots',model_name+'_efficient_batchs'+str(cfg['fine_tune_batch_size'])+'_ep'+str(cfg['fine_tune_train_epoch'])+'','K-'+str(K_name),'lr'+lr_name,'seed'+str(cfg['seed'])  )  #   

    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    
    # seed
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])



    train_transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),

    ])

    # train_transform_aug = preprocess
    train_transform_no_aug = preprocess



    

    if cfg['is_train'] == 1:
        sys.stdout = Logger( os.path.join(cache_dir,'log_train.txt') ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        print("augment transform:")
        for transform in train_transform_aug.transforms:
            print(transform)



        # dataset
        few_shot_dataset = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size = cfg['fine_tune_batch_size']

        val_loader = build_data_loader(data_source=few_shot_dataset.val, batch_size=batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)
        test_loader = build_data_loader(data_source=few_shot_dataset.test, batch_size=batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        train_loader = build_data_loader(data_source=few_shot_dataset.train_x, batch_size=batch_size, tfm=train_transform_aug, is_train=True, shuffle=True)
        
        
        cfg['classnames'] = few_shot_dataset.classnames # 



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of classnames: {len(cfg['classnames'])}")
        print(f"K: {cfg['K']}")
        print(f"use_negative_prompt: {cfg['use_negative_prompt']}")
        if cfg['use_negative_prompt']:
            print(f"negative_template: {cfg['negative_template']}")
            print(f"lambda_neg: {cfg['lambda_neg']}, lambda_margin: {cfg['lambda_margin']}, neg_margin: {cfg['neg_margin']}")
            print(f"alpha_neg: {cfg['alpha_neg']}, neg_loss_scale: {cfg['neg_loss_scale']}")
            print(f"use_conf_gate: {cfg['use_conf_gate']}, conf_gate_gamma: {cfg['conf_gate_gamma']}, conf_gate_min: {cfg['conf_gate_min']}")
            print(f"margin_correct_only: {cfg['margin_correct_only']}")
            print(f"use_class_adaptive_margin: {cfg['use_class_adaptive_margin']}")


        # Model
        model = CustomCLIP(cfg, cfg['classnames'], clip_model ,cfg['template'], False, 16, cfg['csc'])
        model = torch.nn.DataParallel(model).to(device)
        

        for name, param in model.named_parameters():
            if  "prompt_learner" in name : #  
                param.requires_grad_(True)
                print(name)
            else:
                param.requires_grad_(False)

        class_margins = None
        if cfg['use_negative_prompt'] and cfg.get('use_class_adaptive_margin', False):
            # Use non-augmented training images to estimate stable class prototypes.
            proto_loader = build_data_loader(
                data_source=few_shot_dataset.train_x,
                batch_size=batch_size,
                tfm=train_transform_no_aug,
                is_train=False,
                shuffle=False
            )
            class_margins = compute_class_adaptive_margins(
                model=model,
                data_loader=proto_loader,
                num_classes=len(cfg['classnames']),
                cfg=cfg,
                device=device
            )


        criterion = torch.nn.CrossEntropyLoss()
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=5e-4 ,momentum=0.9 ,dampening=0 ,nesterov=False)  # 
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  cfg['fine_tune_train_epoch'] * len(train_loader))

        print(optimizer)
        print(scheduler)

        train_epoch = cfg['fine_tune_train_epoch']

        for train_idx in range(train_epoch):
            correct_samples, all_samples = 0, 0
            loss_list = []
            loss_global_list = []
            loss_neg_list = []
            loss_margin_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))

            model.train()

            if 'RN' in cfg['backbone']:
                model.module.image_encoder.eval()
                
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                outputs = model(images)

                if cfg['use_negative_prompt']:
                    logits, _, logits_neg, _ = outputs

                    # Original FA loss: keep the forced-positive prompt training objective unchanged.
                    loss_global = criterion(logits, target)

                    # Negative prompt supervision. For class y, "not y" should be 0,
                    # and "not other classes" should be 1.
                    class_num = len(cfg['classnames'])
                    neg_targets = build_negative_targets(target, class_num)
                    logits_neg_for_loss = (logits_neg[:, :class_num] / cfg['neg_loss_scale']).float()

                    # Per-sample negative BCE. We keep a mean over classes first, then optionally
                    # apply a conservative confidence gate over samples.
                    loss_neg_per_sample = F.binary_cross_entropy_with_logits(
                        logits_neg_for_loss,
                        neg_targets.float(),
                        reduction='none'
                    ).mean(dim=1)

                    # Margin constraint: the positive score of the true class should be larger
                    # than the negative score "not true class" by at least neg_margin.
                    batch_size = logits.shape[0]
                    logits_pos_temp = logits.view(batch_size, 1 + cfg['K'], class_num)
                    just_forced_logits = logits_pos_temp[:, 0, :]
                    pos_true = just_forced_logits.gather(1, target.view(-1, 1)).squeeze(1) / cfg['neg_loss_scale']
                    neg_true = logits_neg.gather(1, target.view(-1, 1)).squeeze(1) / cfg['neg_loss_scale']

                    if cfg.get('use_class_adaptive_margin', False) and class_margins is not None:
                        margin_y = class_margins.gather(0, target).to(pos_true.device).float()
                    else:
                        margin_y = torch.full_like(pos_true.float(), float(cfg['neg_margin']))

                    loss_margin_per_sample = F.relu(margin_y + neg_true.float() - pos_true.float())

                    if cfg['use_conf_gate']:
                        # Soft lower-bound gate. Unlike p_true^gamma alone, this does not erase
                        # the negative supervision in early few-shot training.
                        with torch.no_grad():
                            prob_true = F.softmax(just_forced_logits.detach() / cfg['neg_loss_scale'], dim=1) \
                                .gather(1, target.view(-1, 1)).squeeze(1)
                            gate = cfg['conf_gate_min'] + (1.0 - cfg['conf_gate_min']) * prob_true.pow(cfg['conf_gate_gamma'])
                    else:
                        gate = torch.ones_like(loss_neg_per_sample)

                    if cfg['margin_correct_only']:
                        with torch.no_grad():
                            pred_train = just_forced_logits.detach().argmax(dim=1)
                            correct_mask = pred_train.eq(target).float()
                    else:
                        correct_mask = torch.ones_like(loss_margin_per_sample)

                    loss_neg = (loss_neg_per_sample * gate).mean()
                    loss_margin = (loss_margin_per_sample * gate * correct_mask).sum() / (gate * correct_mask).sum().clamp_min(1.0)

                    loss = loss_global + cfg['lambda_neg'] * loss_neg + cfg['lambda_margin'] * loss_margin

                    loss_global_list.append(loss_global.item())
                    loss_neg_list.append(loss_neg.item())
                    loss_margin_list.append(loss_margin.item())
                else:
                    logits, _ = outputs
                    loss = criterion(logits, target)

                acc = cls_acc(output=logits , target=target, topk=1)

                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, train_acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
            if cfg['use_negative_prompt']:
                print('    global_loss: {:.4f}, neg_loss: {:.4f}, margin_loss: {:.4f}'.format(
                    sum(loss_global_list)/len(loss_global_list),
                    sum(loss_neg_list)/len(loss_neg_list),
                    sum(loss_margin_list)/len(loss_margin_list)
                ))



            # test
            if  train_idx == train_epoch - 1 :   #  (train_idx + 1) % 100 == 0 or  train_idx == 0  or 
                model.eval()
                with torch.no_grad():
                    correct_samples, all_samples = 0, 0
                    topk = 1

                    for images, labels in tqdm(test_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)[0]

                        outputs = outputs / 100.0
                        batch_size, repeat_len = outputs.shape
                        outputs_temp = outputs.view(batch_size, 1+cfg['K'], len(cfg['classnames']))
                        just_forced_outputs = outputs_temp[:,0,:]


                        pred = just_forced_outputs.topk(topk, 1, True, True)[1].t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                        correct_samples += acc_num
                        all_samples += labels.shape[0]

                    test_acc = correct_samples / all_samples
                    print(f'test Epoch [{train_idx+1}/{train_epoch}], Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')


                    # save model
                    torch.save(model.state_dict(), os.path.join(cfg['cache_dir'], 'model.pth'))

            
    else:
        test_file_name = f'log_test_ood_{cfg["ood_dataset"]}.txt'
        sys.stdout = Logger( os.path.join(cache_dir, test_file_name) ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        print("augment transform:")
        for transform in train_transform_no_aug.transforms:
            print(transform)

        # test ood

        # id dataset
        few_shot_dataset = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size = cfg['test_batch_size']

        test_loader_id = build_data_loader(data_source=few_shot_dataset.test, batch_size=batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        cfg['classnames'] = few_shot_dataset.classnames  #



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of classnames: {len(cfg['classnames'])}")
        print(f"K: {cfg['K']}")
        print(f"use_negative_prompt: {cfg['use_negative_prompt']}")
        if cfg['use_negative_prompt']:
            print(f"negative_template: {cfg['negative_template']}")
            print("negative prompt is used as training-only auxiliary supervision.")
            print("Evaluation keeps the original FA score functions: MCM and GL-MCM.")
            print(f"use_class_adaptive_margin: {cfg['use_class_adaptive_margin']}")


        # ood dataset
        ood_dataset_name = cfg['ood_dataset']

        if ood_dataset_name == 'challenging':
            ood_dataset_name = ['OpenImage_O','NINCO','imagenet-o' ]
        elif ood_dataset_name == 'all':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd','OpenImage_O','NINCO','imagenet-o']
        elif ood_dataset_name == 'nearood':
            ood_dataset_name = ['ssb_hard','NINCO']
        elif ood_dataset_name == 'common':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd']
        else:
            ood_dataset_name = [ood_dataset_name]


        print(f"ood_dataset_name: {ood_dataset_name}")

        # Model
        model = CustomCLIP(cfg, cfg['classnames'], clip_model ,cfg['template'], False, 16, cfg['csc'])
        model = torch.nn.DataParallel(model).to(device)


        # load model
        model.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], 'model.pth')))

        model.eval()
        with torch.no_grad():
            correct_samples, all_samples = 0, 0
            correct_samples_gl, all_samples_gl = 0, 0
            topk = 1
            T = 1.0

            # logit_result
            id_conf = np.array([])
            id_conf_gl = np.array([])


            # run id dataset
            for images, labels in tqdm(test_loader_id):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                if cfg['use_negative_prompt']:
                    logits_id, logits_id_local, _, _ = outputs
                else:
                    logits_id, logits_id_local = outputs

                logits_id /= 100.0
                logits_id_local /= 100.0

                # ----
                batch_size, token_len, repeat_len = logits_id_local.shape
                logits_id_temp = logits_id.view(batch_size, 1+cfg['K'], len(cfg['classnames']))
                just_forced_logits_id = logits_id_temp[:,0,:]


                logits_id_local_temp = logits_id_local.view(batch_size, token_len, 1+cfg['K'], len(cfg['classnames']))
                just_forced_logits_id_local = logits_id_local_temp[:,:,0,:]
                # ----


                smax_global = F.softmax(logits_id/T, dim=-1).cpu().numpy()
                mcm_global_score = np.max(smax_global, axis=1)

                smax_local = F.softmax(logits_id_local/T, dim=-1).cpu().numpy()
                mcm_local_score = np.max(smax_local, axis=(1, 2))
                # print(logits_id.size)

                id_conf = np.concatenate((id_conf, mcm_global_score))
                id_conf_gl = np.concatenate((id_conf_gl, mcm_global_score + mcm_local_score))

                # caculate accuracy
                pred = just_forced_logits_id.topk(topk, 1, True, True)[1].t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples += acc_num
                all_samples += labels.shape[0]

                logits_id_local_gl = just_forced_logits_id_local.max(dim=1)[0] + just_forced_logits_id
                pred_gl = logits_id_local_gl.topk(topk, 1, True, True)[1].t()
                correct_gl = pred_gl.eq(labels.view(1, -1).expand_as(pred_gl))
                acc_num_gl = float(correct_gl[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples_gl += acc_num_gl
                all_samples_gl += labels.shape[0]

            
            test_acc = correct_samples / all_samples
            test_acc_gl = correct_samples_gl / all_samples_gl
            print(f'Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')
            print(f'Test gl accuracy of the model on the test images: { 100 * test_acc_gl :.2f}%')

            # run ood dataset

            # avarage ood 
            avg_auroc, avg_aupr, avg_fpr = 0.0, 0.0, 0.0
            avg_auroc_gl, avg_aupr_gl, avg_fpr_gl = 0.0, 0.0, 0.0

            for cur_ood_dataset in ood_dataset_name:
                print(f'cur ood dataset: {cur_ood_dataset}')
                ood_conf = np.array([])
                ood_conf_gl = np.array([])

                if  cur_ood_dataset == 'ood':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=train_transform_no_aug)
                elif cur_ood_dataset in ['iNaturalist','SUN','Places','OpenImage_O','imagenet-o','ssb_hard']:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'dtd':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'images'), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'NINCO':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'NINCO_OOD_classes'), transform=train_transform_no_aug)
                else:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'test'), transform=train_transform_no_aug)
                
                ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg['test_batch_size'],
                                                            shuffle=False, num_workers=8)


                for images, _ in tqdm(ood_loader):
                    images = images.to(device)

                    outputs = model(images)
                    if cfg['use_negative_prompt']:
                        logits_ood, logits_ood_local, _, _ = outputs
                    else:
                        logits_ood, logits_ood_local = outputs

                    logits_ood /= 100.0
                    logits_ood_local /= 100.0
                    # print(logits_ood.shape)
                    # print(logits_ood_local.shape)

                    smax_global = F.softmax(logits_ood/T, dim=-1).cpu().numpy()
                    mcm_global_score = np.max(smax_global, axis=1)

                    smax_local = F.softmax(logits_ood_local/T, dim=-1).cpu().numpy()
                    mcm_local_score = np.max(smax_local, axis=(1, 2))
                    # print(logits_ood.size)

                    ood_conf = np.concatenate((ood_conf, mcm_global_score))
                    ood_conf_gl = np.concatenate((ood_conf_gl, mcm_global_score + mcm_local_score))

                print('id_conf size',id_conf.size)
                print('ood_conf size',ood_conf.size)

                # ood detection
                tpr = 0.95


                auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
                avg_auroc += auroc
                avg_aupr += aupr
                avg_fpr += fpr
                print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR(0.95): {fpr:.4f}')


                auroc_gl, aupr_gl, fpr_gl = get_measures(id_conf_gl, ood_conf_gl, tpr)
                avg_auroc_gl += auroc_gl
                avg_aupr_gl += aupr_gl
                avg_fpr_gl += fpr_gl
                print(f'AUROC_glmcm: {auroc_gl:.4f}, AUPR_glmcm: {aupr_gl:.4f}, FPR(0.95)_glmcm: {fpr_gl:.4f}')

            
            print('--------------------------------')
            avg_auroc /= len(ood_dataset_name)
            avg_aupr /= len(ood_dataset_name)
            avg_fpr /= len(ood_dataset_name)
            print(f'Average AUROC: {avg_auroc:.4f}, Average AUPR: {avg_aupr:.4f}, Average FPR(0.95): {avg_fpr:.4f}')

            avg_auroc_gl /= len(ood_dataset_name)
            avg_aupr_gl /= len(ood_dataset_name)
            avg_fpr_gl /= len(ood_dataset_name)
            print(f'Average AUROC_glmcm: {avg_auroc_gl:.4f}, Average AUPR_glmcm: {avg_aupr_gl:.4f}, Average FPR(0.95)_glmcm: {avg_fpr_gl:.4f}')
            
if __name__ == '__main__':
    main()
