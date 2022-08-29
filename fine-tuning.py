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
device = torch.device('cuda:0')

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


def training_vali_seperation(training_seq, training_label):
    tra_seq, validation_seq, tra_label, validation_label = train_test_split(training_seq, training_label,
                                                                            test_size = 0.2,
                                                                            stratify = training_label, random_state=521)
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word, attention):
        _, cls = self.BERT(word, attention, return_dict=False)
        x = self.fc_seqn(cls)
        x = self.out(x)
        x = self.sigmoid(x)
        return x

def init_kaiming(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(m.weight, a=0.001, mode='fan_in', nonlinearity='leaky_relu')
        print(m)


from torch.optim.lr_scheduler import MultiStepLR
def Training_process(net, epochs):
    net = copy.deepcopy(net)
    net = net.to(device)
    for name, parm in net.named_parameters():
        print(name)
        if 'fc_seqn' in name:
            net.fc_seqn.apply(init_kaiming)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-6, weight_decay=1.5e-5)
    scheduler = MultiStepLR(optimizer, milestones=[2, 6], gamma=0.25)
    
    print("\n\nStarting training loop...\n\n")
    start_time = time.perf_counter()
    loss_tally = []
    min_valid_loss = np.inf
    best_accuracy = 0.0
    save_acc = 0.0
    for epoch in range(epochs):
        print("begin", epoch, "epochs")
        running_loss= 0.0
        train_loss = 0.0
        running_accuracy = 0.0
        total = 0
        net.train()
        for i, data in enumerate(train_data_loader):
            word, att, label = data
            if i % 500 == 0:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
            word = word.to(device)
            att = att.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(word, att)
            label = label.unsqueeze(1)
            label = label.to(torch.float32)
            outputs = outputs.to(torch.float32)
            loss = criterion(outputs, label)
            train_loss += loss.item()
            predicted = (outputs > 0.5).type(torch.uint8)
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                avg_loss = running_loss / float(50)
                loss_tally.append(avg_loss)
                print(
                    "[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]       loss %.4f   " % (
                        epoch + 1, epochs, i + 1, elapsed_time, avg_loss))
                running_loss = 0.0
        valid_loss = 0.0
        running_accuracy = 0.0
        net.eval()
        for i, data in enumerate(validation_data_loader):
            word, att, label = data
            word = word.to(device)
            att = att.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(word, att)
            label = label.unsqueeze(1)
            label = label.to(torch.float32)
            total += len(label)
            outputs = outputs.to(torch.float32)
            loss = criterion(outputs, label)
            valid_loss += loss.item()
            predicted = (outputs > 0.5)
            running_accuracy += (predicted == label).sum().item()
        print('Epoch Summary')
        print(f'Epoch {epoch + 1} \t\t Training Loss: { train_loss / len(train_data_loader)} \t\t Validation Loss: { valid_loss / len(validation_data_loader)}'
              f'\t\t Accuracy: { running_accuracy / total}')
        if running_accuracy > save_acc:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            save_acc = running_accuracy
            # Saving State Dict
            torch.save(net.state_dict(), 'Bestmodel_.pth')
        scheduler.step()
    print("\nFinished Training\n")
    plt.figure(figsize=(10, 5))
    plt.title("Loss vs. Iterations")
    plt.plot(loss_tally)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("Loss.png")
    plt.show()


def test_process(net, epochs):
    net.load_state_dict(torch.load(r'Bestmodel_.pth'))
    net.eval()
    net = net.to(device)
    criterion = nn.BCELoss()
    print("\n\nStarting testing loop...\n\n")
    start_time = time.perf_counter()
    elapsed_time = 0.0
    loss_tally = []
    gt = []
    pred = []
    test_accuracy = 0
    total = 0
    with torch.no_grad():
        for epoch in range(epochs):
            running_loss= 0.0
            for i, data in enumerate(test_data_loader):
                word, att, label = data
                if i % 500 == 0:
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time
                word = word.to(device)
                att = att.to(device)
                label = label.to(device)
                outputs = net(word, att)
                label = label.unsqueeze(1)
                label = label.to(torch.float32)
                ground_truth = label.cpu()
                ground_truth = ground_truth.tolist()
                gt.append(ground_truth)
                outputs = outputs.to(torch.float32)
                test_predict = (outputs > 0.5)
                test_accuracy += (test_predict == label).sum().item()
                total += len(label)
                predicted = outputs.cpu()
                predicted = predicted.tolist()
                pred.append(predicted)
                loss = criterion(outputs, label)
                running_loss += loss.item()
                if i % 500 == 0:
                    avg_loss = running_loss / float(500)
                    loss_tally.append(avg_loss)
                    print(
                        "[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]       loss %.4f   " % (
                            epoch + 1, epochs, i + 1, elapsed_time, avg_loss))
                    running_loss = 0.0

    print("\nFinished Testing\n")
    print("ACC is ",test_accuracy/total )
    gt = sum(gt,[])
    gt = sum(gt, [])
    pred = sum(pred, [])
    pred = np.array(pred)
    #pred = np.rint(pred)
    #pred = np.ravel(pred)
    pred = pred.tolist()
    pred = list(_flatten(pred))
    return pred, gt


if __name__ == '__main__':
    #hyperparameter
    setup_seed(16)
    batch_size = 4
    epoch = 2
    training_seq, training_label, test_seq, test_label = read_dataset()
    training_seq, validation_seq, training_label, validation_label = training_vali_seperation(training_seq, training_label)
    #import model
    BERT = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    token_training, token_validation, token_test = tokenize_sequence(tokenizer, training_seq,validation_seq, test_seq)

    train_dataset = dataloader(token_training, training_label)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

    val_dataset = dataloader(token_validation, validation_label)
    validation_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

    test_dataset = dataloader(token_test, test_label)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=0)
    model = Architecture(BERT)

    #Training Loop
    Training_process(model, epoch)
    pred, gt = test_process(model, 1)


    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, auc, f1_score, accuracy_score, recall_score

    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(gt, pred)
    print('fpr',fpr)
    print(type(fpr))
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    pred__ = np.rint(np.array(pred))
    pred__ = np.ravel(pred__)
    pred__ = list(pred__)
    print("AUC", roc_auc_score(gt, pred))
    print("F1", f1_score(gt, pred__))
    print("Acc", accuracy_score(gt, pred__))
    print("Recall", recall_score(gt, pred__))

    confusion_mat_1 = confusion_matrix(gt, pred__)
    class_list = ['negative', 'positive']
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_1, display_labels=class_list)
    disp.plot(
        include_values=True,
        cmap="viridis",
        ax=None,
        xticks_rotation="horizontal",
        # values_format=".1f"
    )
    pl.xticks(rotation=45)
    plt.savefig("confusion_matrix.jpg")
    plt.show()






