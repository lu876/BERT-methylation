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
import pandas as pd
import csv


# Begin read file
class Read_excel:
    def __init__(self):
        self.location = r'BERT_vec.xlsx'

    def readfile(self):
        data = pd.read_excel(self.location, sheet_name='Sheet1')
        id = data.iloc[:,0].values
        english = data.iloc[:,1].values
        vec = data.iloc[:, 2].apply(lambda x:x.replace('\n', '').replace('\r', ''))
        vec = vec.str.split('\s+')
        vec = vec.values

        for i in range(len(id)):
            if vec[i][0] == '[':
                del vec[i][0]
            if len(vec[i]) == 769:
                if vec[i][768] == ']':
                    del vec[i][768]

        for i in range(len(id)):
            if '[' in vec[i][0]:
                vec[i][0] = vec[i][0].replace('[', '')
            if ']' in vec[i][767]:
                vec[i][767] = vec[i][767].replace(']', '')

        for i in range(len(id)):
            for j in range(len(vec[i])):
                if vec[i][j] == ']':
                    print('what is ', i,j)
                vec[i][j] = float(vec[i][j])
        englishid_to_vec_dic = {}
        id_to_english = {}
        for i in range(len(id)):
            englishid_to_vec_dic[id[i]] = vec[i]
            id_to_english[id[i]] = english[i]
        return englishid_to_vec_dic, id_to_english

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
        vocab, index = {}, 1  # define dictionary
        vocab['<pad>'] = 0  # the first word is <pad>
        vocab_size = len(all_possibility)  # define the size of the dictionary
        for kmer in all_possibility:
            vocab[kmer] = index  # assign value to dictionary
            index += 1
        inverse_vocab = {index: kemer for kemer, index in vocab.items()}  # define a inverse_dictionary
        return vocab, all_possibility, vocab_size,inverse_vocab


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


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())

    engword = Read_excel()
    id_vec, id_eng = engword.readfile()

    kmer_number = 5
    context_length = 8
    cut = Separate(kmer_number)
    dict, all_possiblity, vocab_size, inverse_vocab = cut.build_dict()

    def find_N_similar(word, lookup_tensor_i, dic, inverse_vocab):
        cosine_distance = []
        device = torch.device('cuda:0')
        for i in dic:
            v_i = lookup_tensor_i
            v_j = dic[i]
            v_j = torch.tensor(v_j)
            v_j = v_j.to(device)
            cosine_sim = torch.cosine_similarity(v_i, v_j, dim=1)
            cosine_distance.append([cosine_sim.cpu().detach().item(), i])
        topN_ids = sorted(cosine_distance, key=(lambda x: x[0]),reverse=True)[0:1]
        element = topN_ids[0][1]
        similar_element= inverse_vocab[element]
        english_vector = dic[element]
        return similar_element

    def validation(net,id_vec,id_eng):
        net.load_state_dict(torch.load(r'net_word2vec.pth'))
        net.eval()
        device = torch.device('cuda:0')
        net.to(device)
        dict_DNA_vec = {}
        embedding = net.embeddings
        print('Starting generate DNA vector')
        with torch.no_grad():
            new_DNA_vec = {}
            for i in range(len(all_possiblity)):
                words = dict[all_possiblity[i]]
                lookup_tensor_i = torch.tensor([words], dtype=torch.long).to(device)
                dict_DNA_vec[all_possiblity[i]] = embedding(lookup_tensor_i)
                english_vector = find_N_similar(all_possiblity[i], dict_DNA_vec[all_possiblity[i]], id_vec,id_eng)
                new_DNA_vec[all_possiblity[i]] = english_vector
                print('finish %', i/len(all_possiblity))
        return new_DNA_vec

    def save(new_DNA_vec):
        path = "DNA_Eng.csv"
        with open(path, 'w') as f:
            for key in new_DNA_vec.keys():
                f.write("%s, %s\n" % (key, new_DNA_vec[key]))

    model_1 = Word2Vec(vocab_size, 768, 2*context_length)  # model (how many words, embedding size, window size)
    new_DNA_vec = validation(model_1, id_vec, id_eng)
    save(new_DNA_vec)
    print('Finish')

