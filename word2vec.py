import re
from torch import nn
import itertools as it
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import pandas as pd


# Begin read file
device = torch.device('cuda:0')

def read_input():
    file_dir = r'methylation/train.csv'
    training_df = pd.read_csv(file_dir)
    training_seq = training_df['seq'].values.tolist()
    training_datasize = len(training_seq)  # count how many sequence
    for i in range(0, training_datasize):  # cut a whole sequence in to words
        if 'N' in training_seq[i]:
            training_seq[i] = training_seq[i].replace('N','')
        training_seq[i] = re.findall(r'\w{5}', str(training_seq[i]))
    return training_seq

# define a dictionary
class Separate:
    def __init__(self, kmer_number):
        self.kmer = kmer_number

    def build_dict(self):
        alfa = ['A', 'G', 'C', 'T']
        keywords = list(it.product(alfa, repeat=self.kmer))  # generate all the possibility words
        keywords = sorted(keywords)
        s = ''
        all_possibility = []
        for i in range(len(keywords)):
            all_possibility.append(s.join(keywords[i]))  # form a list to store all words
        #print(all_possibility)
        vocab, index = {}, 1  # define dictionary
        vocab['<pad>'] = 0  # the first word is <pad>
        vocab_size = len(all_possibility)  # define the size of the dictionary
        for kmer in all_possibility:
            vocab[kmer] = index  # assign value to dictionary
            index += 1
        inverse_vocab = {index: kemer for kemer, index in vocab.items()}  # define a inverse_dictionary
        #print(inverse_vocab)
        return vocab, inverse_vocab, vocab_size


# define center words and context words
class def_context():
    def __init__(self,sentence,window):
        self.sentence = sentence  # read a sentence
        self.window = window      # define a window size

    def extract_word(self):
        count = 0                 # for reduce the size of dataset
        self.context = []         # pre-define the context
        self.label = []           # pre-define the center word
        for sentence in self.sentence:  # read a sentence
            #print('Runing', sentence)
            count += 1
            if count > 50000:
                print('The dataset is too large.')
                break
            for idx, word in enumerate(sentence):  # read each word
                # if idx >= self.window and idx <= len(sentence)-self.window-1:
                self.label.append([sentence[idx]])  # This is the center word, aka, each word of a sentence
                small = []
                index = list(range(max(0, idx - self.window), min(len(sentence), idx + 1 + self.window))) # Context idx range
                index.remove(idx)  # context remove the center word
                if idx - self.window < 0:
                    for m in range(abs(idx - self.window)):
                        small.append('<pad>')
                    small_DNA=[sentence[i] for i in index]
                    #print('D', small_DNA[0])
                    for kk in range(len(small_DNA)):
                        small.append(small_DNA[kk])
                    self.context.append(small)

                if idx + 1 + self.window-len(sentence) > 0:
                    small_DNA=[sentence[i] for i in index]
                    #print('D', small_DNA[0])
                    for kk in range(len(small_DNA)):
                        small.append(small_DNA[kk])
                    for m in range(abs(idx + 1 + self.window-len(sentence))):
                        small.append('<pad>')
                    self.context.append(small)

                if idx - self.window >= 0 and idx + 1 + self.window-len(sentence) <= 0:
                    self.context.append([sentence[i] for i in index])  # store the context in a list
                # self.context.append([sentence[i] for i in range(idx-self.window, min(idx+self.window+1)) if i !=idx])

        #print("Context :", self.context , "Target :", self.label)
        #print("Target :", self.label)
        return self.context, self.label


# Construct a data loader
class dataloader():
    def __init__(self, vocab, context, label, vocab_size):
        self.vocab = vocab      # load the dictionary
        self.context = context  # load the context
        self.label = label      # load the center word (label)
        self.vocab_size = vocab_size # load the size of words

    def __len__(self):
        return len(self.context)  # return the size of a batch data

    def __getitem__(self, idx):  # for each batch
        self.onehot = np.zeros(self.vocab_size+1, dtype=np.float32)  # using one-hot encoding to encode label
        context = [self.vocab[word] for word in self.context[idx]]  # using dictionary to find the context words
        label = [self.vocab[word] for word in self.label[idx]]  # using dictionary to find the center word
        id = self.vocab[self.label[idx][0]]  # for each batch, find the center word
        self.onehot[int(str(id))] = 1  # set the one-hot to 1
        #context = context + [0]*(4 - len(context))  # if the size of context word is less than 4,using 0 to fill
        return np.array(context, dtype=np.int), self.onehot


# network structure#snp
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Word2Vec, self).__init__()
        vocab_size += 1  # first word <pad>
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.linear1 = nn.Linear(context_size * embedding_dim, 1024)  # 1 hidden layer
        self.linear2 = nn.Linear(1024, vocab_size)  # output layer Cbow

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(len(inputs), -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return (log_probs)

def run_code_for_training(net, epochs):
    loss_tally = []
    net = net.to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 500 == 0:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                print("\n[epoch:%d, batch:%5d, elapsed_time=%5d secs] loss: %.3f" %
                      (epoch + 1, i + 1, elapsed_time, running_loss / float(500)))
                loss_tally.append(running_loss / float(500))
                running_loss = 0.0
    torch.save(net.state_dict(), 'net_word2vec.pth') #origin net1
    return loss_tally


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    kmer_number = 5
    data = read_input()
    cut = Separate(kmer_number)
    dicts, inverse_vocab, vocab_size = cut.build_dict()
    context_length = 8
    test = def_context(data, context_length)
    context, label = test.extract_word()
    print(len(context))
    print("Data size", len(label))

    train_dataset = dataloader(dicts, context, label, vocab_size)
    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size=500,
                                                    shuffle=True,
                                                    num_workers=0)

    model_1 = Word2Vec(vocab_size, 768, 2*context_length)  # model (how many words, embedding size, window size)
    loss_1 = run_code_for_training(model_1, 10)  # training

