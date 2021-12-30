import pickle
import numpy as np

import torch
from transformers import BertTokenizer

import config

import os

class DocumentDataset:
    def __init__(self, dataset_split, num_of_labels):

        if dataset_split != "train" and dataset_split != "val" and dataset_split != "test":
            print('Dataset split: ',dataset_split,' is not valid. Exiting..')


        """
        Loading the dataset with specific dataset split
        """
        with open(config.document_dataset_file_path, 'rb') as handle:
            self.dataset = pickle.load(handle)[dataset_split]



        """
        Making Bert Tokenizer
        """
        self.tokenizer   = BertTokenizer.from_pretrained('bert-base-uncased')



        self.num_of_labels  = num_of_labels

        """
        Loading the label indices map
        """
        with open(config.label_indices_map_path, 'rb') as handle:
            self.labels_map = pickle.load(handle)

        """
        Processing the label embeddings to serve target embeddings 
        """
        self.process_embedding_matrix()


    def __len__(self):
        return len(self.dataset)



    def process_embedding_matrix(self):

        label_matrix = np.zeros( (len(self.dataset), self.num_of_labels), dtype=np.int8)
        

        """
        Loading the label embeddings
        """
        with open(config.labels_embeddings, 'rb') as handle:
            label_embeddings = pickle.load(handle).detach().numpy()


        """
        Filling up the label matrix
        """
        for index, element in enumerate(self.dataset):
            labels = element['eurovoc_concepts']


            for label in labels:
                label_matrix[index][self.labels_map[label]] = 1


        """
        Taking the sum in each label row to get the no. of labels for each document
        """
        each_row_sum = label_matrix.sum(axis=1)

        """
        reshaping to make it a column vector
        """
        each_row_sum = each_row_sum.reshape((len(self.dataset), 1))


        """
        Taking dot product of label_matrix and label embeddings so that one each row of embedding respective to one document has embeddings affected from all the 1 hot labels
        Dividing by totalnos. of labels to normalize them
        """
        self.YEmbed  = np.dot(label_matrix, label_embeddings) / each_row_sum




    def __getitem__(self, idx):
        input_text        = self.dataset[idx]['title'] + self.dataset[idx]['text']

        inputs            = self.tokenizer.encode_plus(input_text, max_length=512, pad_to_max_length=True)

        ids               = inputs["input_ids"]
        mask              = inputs["attention_mask"]
        token_type_ids    = inputs["token_type_ids"]

        target_embeddings = self.YEmbed[idx]

        labels            = self.dataset[idx]['eurovoc_concepts']
        target_tensors    = np.zeros((1, self.num_of_labels))
        for each_label in labels:
            index = self.labels_map[each_label]
            target_tensors[0][index] = 1

        return{
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target_embeds": torch.tensor(target_embeddings, dtype=torch.float),
                "targets":torch.tensor(target_tensors, dtype=torch.long)
                }

















"""
This dataset class is for generating the embeddings.
It doesn't have ground truth labels returning
"""
class EmbeddingDataset:
    def __init__(self,dataset_split):
        
        self.flat_file = config.flat_file.format(dataset_split)

        if not os.path.exists(self.flat_file):
            print("File {} doesn't exists".format(self.flat_file))
            exit()


        with open(self.flat_file, 'rb') as handle:
            self.data = pickle.load(handle)


        self.tokenizer   = BertTokenizer.from_pretrained('bert-base-uncased')


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
       input_text = self.data[idx]

       inputs     = self.tokenizer.encode_plus(input_text, max_length=512, pad_to_max_length=True)

       ids               = inputs["input_ids"]
       mask              = inputs["attention_mask"]
       token_type_ids    = inputs["token_type_ids"]


       return{
               "ids": torch.tensor(ids, dtype=torch.long),
               "mask": torch.tensor(mask, dtype=torch.long),
               "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
               "original_segment": input_text
        }
