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


#1 read Dataset
training_path = r'methylation/train.csv'
test_path = r'methylation/test.csv'
solution_path = r'methylation/solution.csv'
DNA_english_dic_path = r'DNA_Eng.csv'

def match():
    training_df = pd.read_csv(training_path)
    with open(DNA_english_dic_path) as f:
        r = csv.reader(f)
        DNA_english_dic = defaultdict(list)
        for row in r:
            DNA_english_dic[row[0]].append(row[1].strip())
    cpg = training_df['Forward_Sequence'].values.tolist()
    training_seq = training_df['seq'].values.tolist()
    cpg_datasize = len(cpg)
    for i in range(cpg_datasize):  # cut a whole sequence in to words
        cpg[i] = cpg[i].replace('[', '')
        cpg[i] = cpg[i].replace(']', '')
    print(cpg[1999], '\n')
    print(training_seq[1999])
    a = [m.start() for m in re.finditer(cpg[0], training_seq[0])]
    print(a[0])
    print(a[0]+ len(cpg[0]))
match()


def read_dataset():
    training_df = pd.read_csv(training_path)
    with open(DNA_english_dic_path) as f:
        r = csv.reader(f)
        DNA_english_dic = defaultdict(list)
        for row in r:
            DNA_english_dic[row[0]].append(row[1].strip())
    training_seq = training_df['seq'].values.tolist()
    training_label = training_df['Beta'].values.tolist()
    test_df = pd.read_csv(test_path)
    solution_df = pd.read_csv(solution_path)
    test_seq = test_df['seq'].values.tolist()
    test_label = solution_df['Beta'].values.tolist()
    site = training_df['Relation_to_UCSC_CpG_Island'].values.tolist()
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

    return training_seq, training_label, test_seq, test_label, site


def training_vali_seperation(training_seq, training_label):
    tra_seq, validation_seq, tra_label, validation_label = train_test_split(training_seq, training_label,
                                                                            test_size = 0.2,
                                                                            stratify = training_label)
    return tra_seq, validation_seq, tra_label, validation_label


def tokenize_sequence(tokenizer, training_seq,val_sequence, test_sequence):
    training_seq = sum(training_seq, [])
    val_sequence = sum(val_sequence, [])
    test_sequence = sum(test_sequence, [])
    token_training = tokenizer.batch_encode_plus(training_seq,
                                                 max_length=400,
                                                 pad_to_max_length=True,
                                                 truncation=True)
    token_val = tokenizer.batch_encode_plus(val_sequence,
                                             max_length=400,
                                             pad_to_max_length=True,
                                             truncation=True)
    token_test = tokenizer.batch_encode_plus(test_sequence,
                                             max_length=400,
                                             pad_to_max_length=True,
                                             truncation=True)
    return token_training, token_val, token_test


class dataloader():
    def __init__(self, train_token, label, site):
        self.train_token_word = train_token['input_ids']
        self.train_token_att = train_token['attention_mask']
        self.label = label
        self.site = site

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
        site = self.site[idx]
        return word, attention, label, site


class Architecture(nn.Module):
    def __init__(self, BERT):
        super(Architecture, self).__init__()
        self.BERT = BERT
        self.fc1 = nn.Linear(768, 512)
        self.fc_seqn = nn.Sequential(
            nn.Linear(768, 60),
            # nn.BatchNorm1d(60),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.001),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word, attention):
        _, cls, _, atten = self.BERT(word, attention, return_dict=False)
        x = self.fc_seqn(cls)
        x = self.out(x)
        x = self.sigmoid(x)
        return x, atten


if __name__ == '__main__':
    #hyperparameter
    batch_size = 4
    training_seq, training_label, test_seq, test_label,site = read_dataset()
    print(site[0])
    training_seq, validation_seq, training_label, validation_label = training_vali_seperation(training_seq, training_label)
    #import model
    BERT = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions = True)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    token_training, token_validation, token_test = tokenize_sequence(tokenizer, training_seq,validation_seq, test_seq)

    test_dataset = dataloader(token_test, test_label, site)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         num_workers=0)
    model = Architecture(BERT)

    model.load_state_dict(torch.load(r'Bestmodel_.pth', map_location='cuda:0'))
    model.eval()
    device = torch.device('cuda:0')
    model = model.to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            word, att,label, site = data
            #print('check out',site)
            word = word.to(device)
            att = att.to(device)
            pred, atten = model(word, att)
            t = 0
            if label == torch.tensor(1) and pred > 0.5:
                total += 1
                y = tuple(t.detach().cpu() for t in atten)
                attention = torch.zeros_like(y[11][0, 0, :,0])
                for m in range(12):
                    attention = y[11][0, m, :,0]+ attention
                t= attention[200]
                #print(t)
                z = 0
                for j in range(400):
                    z = z + attention[j]
                if t > (z)/400:
                    correct += 1
                attention_score = y[11][0, 0, :,0]+y[11][0, 1, :,0]+y[11][0, 2, :,0]+y[11][0, 3, :,0]+y[11][0, 4, :,0]+y[11][0, 5, :,0]+y[11][0, 6, :,0]+y[11][0, 7, :,0]+y[11][0,8, :,0]+y[11][0, 9, :,0]+y[11][0, 10, :,0]+y[11][0, 11, :,0]
                plt.autoscale(True)
                plt.cla()
                plt.bar(range(0, 400), attention_score, width=1)
                plt.legend()
                plt.savefig('pic-{}.png'.format(i))

    print('total is', total)
    print('correct is', correct)









