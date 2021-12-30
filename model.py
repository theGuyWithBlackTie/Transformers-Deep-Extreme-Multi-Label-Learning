from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

import config

class DXML(nn.Module):
    def __init__(self, dropout=0.1):
        super(DXML, self).__init__()
        self.bert    = BertModel.from_pretrained('bert-base-uncased')
        self.W1      = nn.Linear(768, 300)
        self.RELU    = nn.ReLU()
        self.W2      = nn.Linear(300, 100)


        """
        New experiment layers start here
        """
        #self.W1      = nn.Linear(768, 500)
        #self.W3      = nn.Linear(500, 300)



        self.dropout = nn.Dropout(dropout)


    def forward(self, ids, mask, token_type_ids):
        _, output_layer = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids, return_dict=False)

        output          = F.normalize(output_layer, dim=1, p=2) # This is a L2 normalization

        output          = self.W1(output)
        output          = self.RELU(output)
        output          = self.dropout(output)



        """
        New layer added here for experiment
        """
        #output          = self.W3(output)
        #output          = self.RELU(output)
        #output          = self.dropout(output)



        output          = self.W2(output)
        output          = self.dropout(output)

        output          = F.normalize(output, dim=1, p=2) # Removing this might increase the precision score

        return output
