# BERT-methylation
This repo is the implementation of "".
## Environment dependent
- python == 3.6
- matplotlib==3.5.1
- numpy==1.23.2
- openpyxl==3.0.9
- pandas==1.4.3
- scikit_learn==1.1.2
- seaborn==0.11.2
- torch==1.10.2+cu102
- transformers==4.17.0
- xgboost==1.6.1
## Sequence of executing programs and makefiles:
1. Run word2vec.py:
   The program loads the methylation dataset from the folder and trains a Word2Vec network. The program will generate the file "net_word2vec.pth".
2. Run extract word.py:
   This program will read the vocabulary used during BERT training and load the pre-trained BERT framework. The program filters English characters, edits the input format, and outputs a vector corresponding to English. This file will write the output to BERT_vec.xlsx.
3. Run compare.py:
   The program will read the BERT_vec.xlsx, extracting a vector for each English word. Load the trained Word2Vec model parameters, input all possibilities of 6 mers into the Word2Vec model to compute the vector representation of each 6 mer. Finally, DNA_Eng.csv is generated by calculating the cosine similarity between each 6 mer and each English word vector.
4. Run fine-tuning.py:
   This program is a fine-tuning program that reads a DNA methylation dataset. Generate a dictionary of 6 mers and English words from the DNA_Eng.csv file. Convert the training dataset to English words. The trained network parameters are saved in BERTnet.pth.
5. Tsne_show.py:
   Load the fine-tuned model and validation set data, and compute the new vector representation of the validation set. The distribution map of the validation set is drawn by the T-SNE dimensionality reduction method.
