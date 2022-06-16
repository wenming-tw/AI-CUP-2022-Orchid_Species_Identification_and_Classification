import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from torchmetrics import functional as F 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.functional import cosine_similarity

class ImageDataset(Dataset):
    def __init__(self, fold_x, data_dir, is_train):
        self.fold_x = fold_x
        self.data_dir = data_dir
        self.is_train = is_train
        
        self.tr_transform = A.Compose([
            A.Resize(400, 400),
            A.GaussianBlur(blur_limit=(3, 3), p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5), 
                A.FancyPCA(p=0.5), 
                A.HueSaturationValue(p=0.5)], p=1),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomGamma(p=0.5)], p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5), 
            A.RandomCrop(height=384, width=384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.va_transform = A.Compose([
            A.Resize(400, 400),
            A.CenterCrop(height=384, width=384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        df_label = pd.read_csv(os.path.join(data_dir, 'label.csv'))
        self.label_dict = dict(zip(df_label.filename.tolist(), df_label.category.tolist()))

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.fold_x[index])
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #if image.shape[0] < image.shape[1]:
        #    image = np.rot90(image)
        if self.is_train:
            image = self.tr_transform(image=image)['image']
        else:
            image = self.va_transform(image=image)['image']

        label = torch.LongTensor([self.label_dict[self.fold_x[index]]])

        return image, label

class ByolDataset(Dataset):
    def __init__(self, fold_x, data_dir):
        self.fold_x = fold_x
        self.data_dir = data_dir
        self.transform = A.Compose([
            A.Resize(440, 440),    
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.fold_x[index])
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']

        return image

class EarlyStopping():
    def __init__(self, rounds=-1, metric='loss'):
        self.rounds = rounds
        self.count = 0
        self.iter = 0        
        self.metric = list(map(lambda x:x.strip(), metric.split(',')))
        self.metrics = {
            'loss':1e9,
            'acc':0, 
            'rec':0,
            'pre':0,
            'f1':0
        }
        self.flag = False

    def update(self, metrics, iter):
        if self.metric[0] == 'loss' and metrics[self.metric[0]] < self.metrics[self.metric[0]]:
            self.metrics = metrics
            self.count = 0
            self.iter = iter
        elif sum([metrics[m] for m in self.metric]) > sum([self.metrics[m] for m in self.metric]):
            self.metrics = metrics
            self.count = 0     
            self.iter = iter           
        else:
            self.count += 1
        
        if self.count >= self.rounds:
            self.flag = True
        else:
            self.flag = False
    
    def is_early_stop(self):
        return self.flag

def rand5fold(data_dir, seed=0):
    df_label = pd.read_csv(os.path.join(data_dir, 'label.csv'))

    category_set = df_label.category.unique()
    category_dict = {cat:df_label[df_label.category == cat].filename.tolist() for cat in category_set}

    train_0, valid_0, train_1, valid_1, train_2, valid_2, train_3, valid_3, train_4, valid_4 = [], [], [], [], [], [], [], [], [], []
    for key, value in category_dict.items():
        random.Random(seed).shuffle(value)
        l = int(len(value)*0.2)
        
        train_0.extend(value[l:])
        valid_0.extend(value[:l])

        train_1.extend(value[:l]+value[l*2:])
        valid_1.extend(value[l:l*2])

        train_2.extend(value[:l*2]+value[l*3:])
        valid_2.extend(value[l*2:l*3])

        train_3.extend(value[:l*3]+value[l*4:])
        valid_3.extend(value[l*3:l*4])

        train_4.extend(value[:l*4])
        valid_4.extend(value[l*4:])

    random.Random(seed).shuffle(train_0)
    random.Random(seed).shuffle(valid_0)
    random.Random(seed).shuffle(train_1)
    random.Random(seed).shuffle(valid_1)
    random.Random(seed).shuffle(train_2)
    random.Random(seed).shuffle(valid_2)
    random.Random(seed).shuffle(train_3)
    random.Random(seed).shuffle(valid_3)
    random.Random(seed).shuffle(train_4)
    random.Random(seed).shuffle(valid_4)

    return train_0, valid_0, train_1, valid_1, train_2, valid_2, train_3, valid_3, train_4, valid_4

def cal_metrics(preds, target, num_classes, object_mse):
    if object_mse:
        loss = IndexMSELoss(num_classes)(preds, target).item()
    else:
        loss = torch.nn.CrossEntropyLoss()(preds, target).item()
    acc = F.accuracy(preds, target, num_classes=num_classes).item()
    rec = F.recall(preds, target, num_classes=num_classes, average='macro').item()
    pre = F.precision(preds, target, num_classes=num_classes, average='macro').item()
    f1 = F.f1(preds, target, num_classes=num_classes, average='macro').item()

    return {'loss':loss, 'acc':acc, 'rec':rec, 'pre':pre, 'f1':f1}

class IndexMSELoss(torch.nn.Module):
    def __init__(self, num_classes, pos_mean=3, pos_std=0.2, neg_mean=0, neg_std=0.2, reduction='mean'):
        super(IndexMSELoss, self).__init__()
        self.num_classes = num_classes
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.neg_mean = neg_mean
        self.neg_std = neg_std
        self.reduction = reduction

    def forward(self, input, target):
        tmp_target = torch.empty(input.shape[0], self.num_classes, device=input.device).normal_(mean=0, std=0.2)
        index = (torch.arange(input.shape[0], dtype=torch.long), target.ravel())
        tmp_target[index] = torch.empty(input.shape[0], device=input.device).normal_(mean=self.pos_mean, std=self.pos_std)
        target = tmp_target
        return torch.nn.functional.mse_loss(input, target, reduction=self.reduction)

def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    index = torch.randperm(x.shape[0]).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cpu'):
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1    

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))
    
    index = torch.randperm(x.shape[0]).to(device)

    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True    