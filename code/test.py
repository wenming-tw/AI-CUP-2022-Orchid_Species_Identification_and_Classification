import os
from xxlimited import Str
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

import logging
import argparse
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ConvNextForImageClassification

import cv2
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    def __init__(self, x):
        self.x = x
       
        self.transform = A.Compose([
            A.Resize(400, 400),
            A.CenterCrop(height=384, width=384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        file = self.x[index]
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)['image']

        return image

def predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch.to(device)
            logits = model(x).logits.cpu()
            preds.append(torch.softmax(logits, dim=-1))
            
        preds = torch.cat(preds, dim=0)
    torch.cuda.empty_cache()
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_dir', default = '../data/orchid_public_set')
    parser.add_argument('--private_dir', default = '../data/orchid_private_set')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bsize', default=128, type=int)
    parser.add_argument('--num_classes', default=219, type=int)
    parser.add_argument('--experiment_id', default='04_30_2022_22_50_18', type=str)
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()
    print(vars(args))
    
    print('experiment_id:', args.experiment_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    os.makedirs('../logs', exist_ok=True)

    outs = []
    for i in range(5):
        model = ConvNextForImageClassification.from_pretrained('facebook/convnext-xlarge-384-22k-1k')
        model.classifier = torch.nn.Linear(model.classifier.in_features, 219)
        model.load_state_dict(torch.load(os.path.join('../checkpoint', args.experiment_id, '{}.ckpt'.format(i))).get('checkpoint'))
        model.to(device)

        public_data = os.listdir(args.public_dir)
        public_dataset = list(map(lambda x:os.path.join(args.public_dir, x), public_data))

        private_data = os.listdir(args.private_dir)
        private_dataset = list(map(lambda x:os.path.join(args.private_dir, x), private_data))
        
        print(len(public_data), len(private_data))

        ## dataset and dataloader
        dataset = ImageDataset(public_dataset + private_dataset)
        data_loader = DataLoader(dataset, batch_size=args.bsize, pin_memory=True, shuffle=False)

        out = predict_model(model, data_loader, device)
        torch.save(out, os.path.join('../outputs', args.experiment_id+'_{}.pt'.format(i)))
        print(out.shape)

        out2label = dict(zip(public_data+private_data, out.argmax(1).tolist()))
        
        df_out = pd.read_csv('../data/submission_template.csv')
        df_out.category = -1
        df_out.category = df_out.filename.apply(lambda x: out2label.get(x, -1))
        df_out.to_csv(os.path.join('../outputs', args.experiment_id+'_{}.csv'.format(i)), index=False)

        outs.append(out)
    outs = torch.stack(outs, dim=0)
    print(outs.shape)

    torch.save(outs, os.path.join('../outputs', args.experiment_id+'.pt'))
    
    out2label = dict(zip(public_data+private_data, outs.mean(0).argmax(1).tolist()))
    
    df_out = pd.read_csv('../data/submission_template.csv')
    df_out.category = -1
    df_out.category = df_out.filename.apply(lambda x: out2label.get(x, -1))
    df_out.to_csv(os.path.join('../outputs', args.experiment_id+'.csv'), index=False)