import os
import csv
import cv2
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--seed', default=2023, type=int, help='manual seed')
    parser.add_argument('--train_mode', default='with_val', type=str, help='with_val, without_val, test')
    parser.add_argument('--task', default='task1', type=str, help='task 1,2,3')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--net_name', default='resnet152', type=str, help='resnet18, resnet50, resnet152')
    parser.add_argument('--best_ckpt_dir', default='best_ckpt', type=str, help='best checkpoint dir')
    parser.add_argument('--load_ckpt_dir', default='task1_100.pt', type=str, help='load best checkpoint dir')
    args = parser.parse_args()
    return args

class TaskDataset(Dataset):
    def __init__(self, data, root, num_classes, args, alphabets2index):
        self.data = [sample for sample in data]
        self.args = args
        self.root = root
        self.num_classes = num_classes
        self.alphabets2index = alphabets2index
        self.transform = T.Compose([
                T.ToTensor(), # [H, W, C] to [C, H, W], and convert to [0,1]
                # 從ImageNet dataset 取得這些平均值與標準差
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                T.RandomRotation(degrees=20),
                T.RandomHorizontalFlip()
            ])
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        path = f"{self.root}/{filename}"
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        label_index = []
        for alphabet in label:
            label_index.append(self.alphabets2index[alphabet])

        label_index = torch.tensor(label_index)
        # one_hot_label_index: [num_preds, 62]
        one_hot_label_index = F.one_hot(label_index, num_classes=self.num_classes).to(torch.float32)

        if self.args.train_mode == 'test':
            return img, filename
        else:
            return img, one_hot_label_index

    def __len__(self):
        return len(self.data)

class Pretrained_model(nn.Module):
    def __init__(self, net_name, args, num_preds, num_classes):
        super(Pretrained_model, self).__init__()
        if net_name == 'resnet18' or net_name == 'resnet50' or net_name == 'resnet152':
            if net_name == 'resnet18':
                net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif net_name == 'resnet50':
                net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif net_name == 'resnet152':
                net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 1024)

        self.pretrained_net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(1024, num_preds*num_classes)
    
    def forward(self, x):
        x = self.pretrained_net(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

def train(num_preds, model, dataloader, optimizer, scheduler, device):
    total_loss = 0
    correct_count = 0
    m = nn.Sigmoid()
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    for data in dataloader:
        model.train()
        # img: [batch_size, 32, 32], one_hot_gt_label: [batch_size, num_preds, 62]
        img, one_hot_gt_label = data[0].to(device), data[1].to(device) 

        pred = model(img) # [batch_size, num_preds*62]
        pred = pred.view(pred.shape[0], num_preds, -1) # [batch_size, num_preds, 62]
        pred_label = pred.argmax(dim=-1) # [batch_size, num_preds]
        gt_label = one_hot_gt_label.argmax(dim=-1) # [batch_size, num_preds]
        correct = torch.all(pred_label == gt_label, dim=-1)
        correct_count += correct.sum().item()

        # m_pred = m(pred.view(-1, pred.shape[-1]))    
        # loss = criterion(m_pred, one_hot_gt_label.view(-1, one_hot_gt_label.shape[-1]))
        m_pred = m(pred.view(pred.shape[0], -1))
        loss = criterion(m_pred, one_hot_gt_label.view(one_hot_gt_label.shape[0], -1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    total_loss /= len(dataloader)
    acc = correct_count / len(dataloader.dataset)
    
    return total_loss, acc

def val(num_preds, model, dataloader, device):
    total_loss = 0
    correct_count = 0
    m = nn.Sigmoid()
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    for data in dataloader:
        model.eval()
        img, one_hot_gt_label = data[0].to(device), data[1].to(device) 

        pred = model(img)
        pred = pred.view(pred.shape[0], num_preds, -1) 
        pred_label = pred.argmax(dim=-1)
        gt_label = one_hot_gt_label.argmax(dim=-1) 
        correct = torch.all(pred_label == gt_label, dim=-1)
        correct_count += correct.sum().item()
        
        # m_pred = m(pred.view(-1, pred.shape[-1]))
        # loss = criterion(m_pred, one_hot_gt_label.view(-1, one_hot_gt_label.shape[-1]))
        m_pred = m(pred.view(pred.shape[0], -1))
        loss = criterion(m_pred, one_hot_gt_label.view(one_hot_gt_label.shape[0], -1))
        total_loss += loss.item()
    
    total_loss /= len(dataloader)
    acc = correct_count / len(dataloader.dataset)

    return total_loss, acc

def test(num_preds, model, dataloader, device, alphabets, csv_writer):
    for img, filenames in dataloader:
        model.eval()
        img = img.to(device)
        pred = model(img)
        pred = pred.view(pred.shape[0], num_preds, -1) 
        pred_label = pred.argmax(dim=-1) # [batch_size, num_preds]
        
        for i in range(len(filenames)):
            alphabet_str = ''
            for j in range(num_preds):
                alphabet = str(alphabets[pred_label[i][j].item()])
                alphabet_str += alphabet

            csv_writer.writerow([filenames[i], alphabet_str])

    print('test finished')

def save_ckpt(ckpt_dir, model, optimizer, scheduler, train_loss, train_acc):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc
    }

    torch.save(state, ckpt_dir)

def load_ckpt(ckpt_dir, model, optimizer, scheduler, device):
    state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    
    print('model loaded from %s' % ckpt_dir)
    return model, optimizer, scheduler

if __name__ == "__main__":
    alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    num_classes=len(alphabets)
    alphabets2index = {alphabet:i for i, alphabet in enumerate(alphabets)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.task == 'task1':
        num_preds = 1
    elif args.task == 'task2':
        num_preds = 2
    elif args.task == 'task3':
        num_preds = 4

    dataset_dir = 'dataset'
    TRAIN_PATH = dataset_dir + "/train"
    TEST_PATH = dataset_dir + "/test"
    if args.train_mode == 'with_val':
        train_data = []
        val_data = []
        with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                # row is [filename, label], ex: ['task3/XXXXXXXXX.png', '25fG']
                if row[0].startswith(args.task):
                    if random.random() < 0.8:
                        train_data.append([row[0], list(row[1])])
                    else:
                        val_data.append([row[0], list(row[1])])
    else:
        train_data = []
        with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                if row[0].startswith(args.task):
                    train_data.append([row[0], list(row[1])])
        test_data = []
        with open(f'{dataset_dir}/sample_submission.csv', newline='') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                if row[0].startswith(args.task):
                    test_data.append([row[0], list(row[1])])
    # --------- load a dataset ------------------------------------
    train_dataset = TaskDataset(train_data, TRAIN_PATH, num_classes, args, alphabets2index)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, \
                    shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.train_mode == 'with_val':
        val_dataset = TaskDataset(val_data, TRAIN_PATH, num_classes, args, alphabets2index)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, \
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        test_dataset = TaskDataset(test_data, TEST_PATH, num_classes, args, alphabets2index)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, \
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # --------- init model ------------------------------------
    model = Pretrained_model(args.net_name, args, num_preds, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( \
        optimizer=optimizer, T_max=args.epoch_size, eta_min=0)

    best_dir_path = args.best_ckpt_dir
    os.makedirs(best_dir_path, exist_ok=True)
    if args.train_mode == 'test':
        load_ckpt_dir = best_dir_path + '/' + args.load_ckpt_dir
        model, optimizer, scheduler = load_ckpt(load_ckpt_dir, model, optimizer, scheduler, device)
    # --------- training loop ------------------------------------
    step = 10
    iters = args.epoch_size // step
    if args.train_mode == 'with_val':
        for i in range(iters):
            for j in tqdm(range(step)):
                train_loss, train_acc = train( \
                    num_preds, model, train_loader, optimizer, scheduler, device)
                val_loss, val_acc = val(num_preds, model, val_loader, device)
            
            epoch = (i+1)*step
            print('[epoch %d] train loss: %.4f' %(epoch, train_loss), "train_acc: %.4f" % train_acc)
            print('[epoch %d] val loss: %.4f' %(epoch, val_loss), "val_acc: %.4f" % val_acc)
    elif args.train_mode == 'without_val':
        for i in range(iters):
            for j in tqdm(range(step)):
                train_loss, train_acc = train( \
                    num_preds, model, train_loader, optimizer, scheduler, device)
            
            epoch = (i+1)*step
            print('[epoch %d] train loss: %.4f' %(epoch, train_loss), "train_acc: %.4f" % train_acc)

        ckpt_dir = f'{best_dir_path}/{args.task}_{epoch}.pt'
        save_ckpt(ckpt_dir, model, optimizer, scheduler, train_loss, train_acc)
    elif args.train_mode == 'test':
        write_csv_path = 'submission.csv'
        if os.path.exists(write_csv_path):
            csv_writer = csv.writer(open(write_csv_path, 'a', newline=''))
        else:
            csv_writer = csv.writer(open(write_csv_path, 'w', newline=''))
            csv_writer.writerow(["filename", "label"])
        
        test(num_preds, model, test_loader, device, alphabets, csv_writer)
