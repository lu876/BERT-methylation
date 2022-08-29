import copy
import torch.optim as optim
import numpy as np
import re
import pandas as pd
import csv
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast
from collections import defaultdict
from tkinter import _flatten
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pylab as pl
import random
device = torch.device('cuda:1')

#1 read Dataset
training_path = r'methylation/train.csv'
test_path = r'methylation/test.csv'
solution_path = r'methylation/solution.csv'
DNA_english_dic_path = r'DNA_Eng.csv'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def read_dataset():
    training_df = pd.read_csv(training_path)
    with open(DNA_english_dic_path) as f:
        r = csv.reader(f)
        DNA_english_dic = defaultdict(list)
        for row in r:
            DNA_english_dic[row[0]].append(row[1].strip())
        print(DNA_english_dic)
    training_seq = training_df['seq'].values.tolist()
    training_label = training_df['Beta'].values.tolist()
    test_df = pd.read_csv(test_path)
    solution_df = pd.read_csv(solution_path)
    test_seq = test_df['seq'].values.tolist()
    test_label = solution_df['Beta'].values.tolist()
    training_datasize = len(training_seq)  # count how many sequence
    test_datasize = len(test_seq)  # count how many sequence
    print("There are", training_datasize, " data")
    hist_train = []
    for i in range(0, training_datasize):  # cut a whole sequence in to words
        if "N" in training_seq[i]:
            training_seq[i] = training_seq[i].replace('N', '')
        training_seq[i] = re.findall(r'\w{5}', str(training_seq[i]))
    for i in range(0, test_datasize):
        if "N" in test_seq[i]:
            test_seq[i] = test_seq[i].replace('N', '')
        test_seq[i] = re.findall(r'\w{5}', str(test_seq[i]))
    for i in range(0, training_datasize):
        for j in range(len(training_seq[i])):
            training_seq[i][j] = DNA_english_dic[training_seq[i][j]]
    for i in range(0, test_datasize):
        for j in range(len(test_seq[i])):
            test_seq[i][j] = DNA_english_dic[test_seq[i][j]]

    for i in range(0, training_datasize):
        hist_train.append(len(training_seq[i]))
        training_seq[i]= list(_flatten(training_seq[i]))
        training_seq[i] = [' '.join(training_seq[i])]  # form a sentence, remove ,
    for i in range(0, test_datasize):
        test_seq[i]= list(_flatten(test_seq[i]))
        test_seq[i] = [' '.join(test_seq[i])]  # form a sentence, remove ,
    print('training_seq', training_seq[0])

    return training_seq, training_label, test_seq, test_label

def tokenize_sequence(tokenizer, training_seq, test_sequence):
    training_seq = sum(training_seq, [])
    test_sequence = sum(test_sequence, [])
    token_training = tokenizer.batch_encode_plus(training_seq,
                                                 max_length=400,
                                                 pad_to_max_length=True,
                                                 truncation=True)
    token_test = tokenizer.batch_encode_plus(test_sequence,
                                             max_length=400,
                                             pad_to_max_length=True,
                                             truncation=True)
    return token_training, token_test


class dataloader():
    def __init__(self, train_token, label):
        self.train_token_word = train_token['input_ids']
        self.train_token_att = train_token['attention_mask']
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        word = self.train_token_word[idx]
        attention = self.train_token_att[idx]
        label = self.label[idx]
        # convert to tensor
        word = torch.tensor(word)
        attention = torch.tensor(attention)
        label = torch.tensor(label)
        return word, attention, label


class Architecture(nn.Module):
    def __init__(self, BERT):
        super(Architecture,self).__init__()
        self.BERT = BERT
        self.fc1 = nn.Linear(768, 512)
        self.fc_seqn = nn.Sequential(
            nn.Linear(768, 60),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.out = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word, attention):
        _, cls = self.BERT(word, attention, return_dict=False)
        x = self.fc_seqn(cls)
        x = self.out(x)
        x = self.sigmoid(x)
        return x, cls.squeeze(0).detach().cpu().tolist()

def init_kaiming(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(m.weight, a=0.001, mode='fan_in', nonlinearity='leaky_relu')
        print(m)


def process(net, data_loader):
    net.load_state_dict(torch.load(r'Bestmodel_.pth'))
    net.eval()
    net = net.to(device)
    print("\n\nStarting testing loop...\n\n")
    feature = []
    gt_label = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            word, att, label = data
            word = word.to(device)
            att = att.to(device)
            outputs, cls = net(word, att)
            label = label.tolist()
            feature.append(cls)
            gt_label.append(label)
    return feature, gt_label


if __name__ == '__main__':
    #hyperparameter
    setup_seed(16)
    batch_size = 1
    training_seq, training_label, test_seq, test_label = read_dataset()
    #import model
    BERT = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    token_training, token_test = tokenize_sequence(tokenizer, training_seq, test_seq)

    train_dataset = dataloader(token_training, training_label)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)

    test_dataset = dataloader(token_test, test_label)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         num_workers=0)
    model = Architecture(BERT)
    training_feature, training_label = process(model, train_data_loader)
    training_feature = np.array(training_feature)
    training_label = np.array(training_label)
    np.save(r'training_feature.npy', training_feature)
    np.save(r'training_label.npy', training_label)

    test_feature, test_label = process(model, test_data_loader)
    test_feature = np.array(test_feature)
    test_label = np.array(test_label)
    np.save(r'test_feature.npy', test_feature)
    np.save(r'test_label.npy', test_label)


