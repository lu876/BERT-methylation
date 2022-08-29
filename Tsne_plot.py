# The 'visualize_layerwise_embeddings' function is adapted from
# https://www.kaggle.com/code/tanmay17061/transformers-bert-hidden-embeddings-visualization

import re
import pandas as pd
import csv
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast
from collections import defaultdict
from tkinter import _flatten
import matplotlib.pyplot as plt
import time
import pylab as pl
from sklearn.manifold import TSNE
import seaborn as sns

device = torch.device('cuda:0')

#1 read Dataset
training_path = r'methylation/train.csv'
test_path = r'methylation/test.csv'
solution_path = r'methylation/solution.csv'
DNA_english_dic_path = r'DNA_Eng.csv'

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
        training_seq[i] = re.findall(r'\w{5}', str(training_seq[i]))
    for i in range(0, test_datasize):
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


def training_vali_seperation(training_seq, training_label):
    tra_seq, validation_seq, tra_label, validation_label = train_test_split(training_seq, training_label,
                                                                            test_size = 0.2,
                                                                            stratify = training_label, random_state=1337)
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
            #nn.BatchNorm1d(60),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word, attention):
        _, cls, hidden_states = self.BERT(word, attention, return_dict=False, output_hidden_states=True)
        x = self.fc_seqn(cls)
        x = self.out(x)
        x = self.sigmoid(x)
        return x, hidden_states


# This function is adapted from
# https://www.kaggle.com/code/tanmay17061/transformers-bert-hidden-embeddings-visualization

def visualize_layerwise_embeddings(hidden_states, masks,ys,epoch,title,layers_to_visualize=[0,1,2,3,4,5,6,7,8,9,10,11]):
    print('visualize_layerwise_embeddings for',title,'epoch',epoch)
    dim_reducer = TSNE(n_components=2)
    num_layers = len(layers_to_visualize)
    fig = plt.figure(figsize=(13,(num_layers/3)*4)) #each subplot of size 3x3
    ax = [fig.add_subplot(int(num_layers/3),3,i+1) for i in range(num_layers)]
    ys = ys.numpy().reshape(-1)
    pl.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    for i,layer_i in enumerate(layers_to_visualize):#range(hidden_states):
        layer_hidden_states = hidden_states[layer_i]
        averaged_layer_hidden_states = torch.div(layer_hidden_states.sum(dim=1), masks.sum(dim=1,keepdim=True))
        layer_dim_reduced_vectors = dim_reducer.fit_transform(averaged_layer_hidden_states.numpy())
        df = pd.DataFrame.from_dict({'x':layer_dim_reduced_vectors[:,0],'y':layer_dim_reduced_vectors[:,1],'label':ys})
        df.label = df.label.astype(int)
        sp = sns.scatterplot(data=df,x='x',y='y',hue='label',ax=ax[i], s=8, legend = 'full')

        plt.setp(sp.get_xticklabels(), fontsize=6)
        plt.setp(sp.get_yticklabels(), fontsize=6)
        #sp.set_xlabel('1st dim', labelpad=2.5, fontsize=6)
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].legend(handles=handles[0:], labels=labels[0:])
        sp.set(xlabel =None)
        sp.set(ylabel=None)
        for lh in sp.legend_.legendHandles:
            lh.set_alpha(1)
            lh._sizes = [5]
        plt.setp(sp.get_legend().get_texts(), fontsize=6)
        plt.setp(sp.get_legend().get_title(), fontsize=6)
        #fig.suptitle(f"{title}: epoch {epoch}")
        ax[i].set_title(f"layer {layer_i+1}", fontsize=8)
    plt.savefig('TSNE-plot')
    print('Done')



def Validation_process(net, epochs):
    net.load_state_dict(torch.load(r'Bestmodel_.pth'))
    net.eval()
    net = net.to(device)
    criterion = nn.BCELoss()
    print("\n\nStarting testing loop...\n\n")
    start_time = time.perf_counter()
    elapsed_time = 0.0
    loss_tally = []
    with torch.no_grad():
        for epoch in range(epochs):
            running_loss= 0.0
            train_masks, train_ys = torch.zeros(0, 400), torch.zeros(0, 1)
            train_hidden_states = None
            for i, data in enumerate(test_data_loader):
                word, att, label = data
                if i % 500 == 0:
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time
                word = word.to(device)
                att = att.to(device)
                label = label.to(device)
                outputs,hidden_states = net(word, att)
                #print('hidden state', hidden_states)
                train_masks = torch.cat([train_masks, att.cpu()])
                train_ys = torch.cat([train_ys, label.cpu().view(-1, 1)])
                if type(train_hidden_states) == type(None):
                    train_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
                else:
                    train_hidden_states = tuple(torch.cat([layer_hidden_state_all, layer_hidden_state_batch.cpu()]) for
                                                layer_hidden_state_all, layer_hidden_state_batch in
                                                zip(train_hidden_states, hidden_states))

                if i % 1000 == 999:
                    avg_loss = running_loss / float(1000)
                    loss_tally.append(avg_loss)
                    print(
                        "[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]" % (
                            epoch + 1, epochs, i + 1, elapsed_time))
                    running_loss = 0.0
                    visualize_layerwise_embeddings(train_hidden_states, train_masks, train_ys, epoch, 'train_data')
                    train_hidden_states = None
                    train_masks, train_ys = torch.zeros(0, 400), torch.zeros(0, 1)

    print("\nFinished Visualization\n")



if __name__ == '__main__':
    #hyperparameter
    batch_size = 1
    epoch = 8
    training_seq, training_label, test_seq, test_label = read_dataset()
    training_seq, validation_seq, training_label, validation_label = training_vali_seperation(training_seq, training_label)
    #import model
    BERT = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    token_training, token_validation, token_test = tokenize_sequence(tokenizer, training_seq,validation_seq, test_seq)
    test_dataset = dataloader(token_test, test_label)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)

    model = Architecture(BERT)
    Validation_process(model, 1)







