import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

import logging
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from utils import ImageDataset, EarlyStopping, IndexMSELoss, rand5fold, cal_metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import ConvNextForImageClassification, SwinForImageClassification

def evaluate_model(model, loader, num_classes, object_mse, device):
    model.eval()
    target, preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            x, y = [b.to(device) for b in batch]
            target.append(y.cpu())
            preds.append(model(x).logits.cpu())
            
        target = torch.cat(target, dim=0).ravel()
        preds = torch.cat(preds, dim=0)
        metrics = cal_metrics(preds, target, num_classes, object_mse)
    torch.cuda.empty_cache()
    return metrics

def train_model(tr_fold, va_fold, args, device, experiment_id, fold):
    print('train:', len(tr_fold), 'valid:', len(va_fold))
    ## initial model
    if 'convnext' in args.model:
        model = ConvNextForImageClassification.from_pretrained('facebook/convnext-xlarge-384-22k-1k')
    elif 'swin' in args.model:
        model = SwinForImageClassification.from_pretrained('microsoft/swin-large-patch4-window12-384-in22k')
    else:
        raise
    model.classifier = torch.nn.Linear(model.classifier.in_features, args.num_classes)
    model.to(device)
    ## dataset and dataloader
    tr_dataset = ImageDataset(tr_fold, args.data_dir, is_train=True)
    va_dataset = ImageDataset(va_fold, args.data_dir, is_train=False)

    tr_loader = DataLoader(tr_dataset, batch_size=args.tr_bsize, pin_memory=True, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=args.va_bsize, pin_memory=True, shuffle=False)
    
    ## objective function and optimizer and lr_scheduler 
    criterion = IndexMSELoss(args.num_classes).to(device) if args.object_mse else torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    
    ## early stopping and tensorboard writter
    early_stopping = EarlyStopping(20, metric='acc,f1')
    writter = SummaryWriter(os.path.join('../tensorboard', experiment_id, 'fold_{}'.format(fold)))

    iter = 0
    eval_iter = len(tr_loader)
    for ep in range(args.epochs):
        for batch in tr_loader:
            iter += 1
            ## train model
            model.train()
            x, y = [b.to(device) for b in batch]
            logit = model(x).logits
            loss = criterion(logit, y.ravel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter % eval_iter == 0:
                ## valid model
                metrics = evaluate_model(model, va_loader, args.num_classes, args.object_mse, device)
                print(' | '.join(['Fold:{} | Epoch:{:03d}, Iteration:{:05d}'.format(fold, ep, iter)] + ['{}:{:.6f}'.format(key, val) for key, val in metrics.items()]))
                
                ## writter tnesorboard
                for key, value in metrics.items():
                    writter.add_scalar(key, value, iter)        
                
                ## early stopping and save model
                early_stopping.update(metrics, iter)
                if early_stopping.is_early_stop():
                    print(' | '.join(['EarlyStopping, Fold:{} | Epoch:{:03d}, Iteration:{:05d}'.format(
                        fold, early_stopping.iter//len(tr_loader), early_stopping.iter)] + ['{}:{:.6f}'.format(key, val) for key, val in early_stopping.metrics.items()]))
                    break
                else:
                    ckpt = early_stopping.metrics.copy()
                    ckpt['iter'] = early_stopping.iter
                    ckpt['checkpoint'] = model.state_dict()
                    torch.save(ckpt, os.path.join('../checkpoint', experiment_id, '{}.ckpt'.format(fold)))
                
                if metrics['acc'] > 0.92:
                    eval_iter = 50
        scheduler.step()
        if early_stopping.is_early_stop():
            break        
       
    return model.cpu(), early_stopping.iter//len(tr_loader), early_stopping.metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = '../data/training')
    parser.add_argument('--model', default='convnext', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tr_bsize', default=8, type=int)
    parser.add_argument('--va_bsize', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--num_classes', default=219, type=int)
    parser.add_argument('--object_mse', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()
    print(vars(args))
    
    experiment_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") 
    print('experiment_id:', experiment_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(
        filename='../logs/{}.log'.format(experiment_id),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info(__file__)
    logger.info(str(vars(args)))
    
    os.makedirs(os.path.join('../checkpoint', experiment_id))
    train_0, valid_0, train_1, valid_1, train_2, valid_2, train_3, valid_3, train_4, valid_4 = rand5fold(args.data_dir, seed=args.seed)

    model_0, epoch_0, metrics_0 = train_model(train_0, valid_0, args, device, experiment_id, 0)
    logger.info(' | '.join(['Epoch:{:03d}'.format(epoch_0)] + ['{}:{:.6f}'.format(key, val) for key, val in metrics_0.items()]))

    model_1, epoch_1, metrics_1 = train_model(train_1, valid_1, args, device, experiment_id, 1)
    logger.info(' | '.join(['Epoch:{:03d}'.format(epoch_1)] + ['{}:{:.6f}'.format(key, val) for key, val in metrics_1.items()]))

    model_2, epoch_2, metrics_2 = train_model(train_2, valid_2, args, device, experiment_id, 2)
    logger.info(' | '.join(['Epoch:{:03d}'.format(epoch_2)] + ['{}:{:.6f}'.format(key, val) for key, val in metrics_2.items()]))

    model_3, epoch_3, metrics_3 = train_model(train_3, valid_3, args, device, experiment_id, 3)
    logger.info(' | '.join(['Epoch:{:03d}'.format(epoch_3)] + ['{}:{:.6f}'.format(key, val) for key, val in metrics_3.items()]))

    model_4, epoch_4, metrics_4 = train_model(train_4, valid_4, args, device, experiment_id, 4)
    logger.info(' | '.join(['Epoch:{:03d}'.format(epoch_4)] + ['{}:{:.6f}'.format(key, val) for key, val in metrics_4.items()]))

    avg_metric = dict()
    for key in metrics_0.keys():
        avg_metric[key] = (metrics_0[key] + metrics_1[key] + metrics_2[key] + metrics_3[key] + metrics_4[key])/5

    logger.info(' | '.join(['Average'] + ['{}:{:.6f}'.format(key, val) for key, val in avg_metric.items()]))