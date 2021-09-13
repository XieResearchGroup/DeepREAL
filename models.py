import numpy as np
#--------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d, Dropout, LogSoftmax
from copy import deepcopy
#--------------------------
from torch_geometric.utils import to_dense_batch
from model_Yang import *
#--------------------------

from resnet import ResnetEncoderModel
#--------------------------

from data_tool_box import *


class DTI_3_class_V3(nn.Module):
    def __init__(self, all_config=None, DTI_binary_pretrained=None):
        super(DTI_3_class_V3,self).__init__()
        self.all_config=all_config
        self.DTI_binary_pretrained = deepcopy(DTI_binary_pretrained)
        self.attentive_interaction_pooler = AttentivePooling(300 )
        self.interaction_pooler = EmbeddingTransform(300 + 256, 128, 64, 0.1)
        choice = 684
        self.multi_classifer = EmbeddingTransform(choice, 64, 3)

    def forward(self,batch_protein_tokenized,batch_chem_graphs,epoch):

        batch_chem_graphs_repr_pooled_binary, batch_protein_repr_resnet_binary, interaction_vector_binary = self.DTI_binary_pretrained.embed3(
                batch_protein_tokenized, batch_chem_graphs)

        ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(batch_chem_graphs_repr_pooled_binary,    batch_protein_repr_resnet_binary) # same as input dimension
        inter_vect3 = self.interaction_pooler(  torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)

        inter_vect3 = torch.cat( (inter_vect3,
                                  interaction_vector_binary,
                                  batch_chem_graphs_repr_pooled_binary[:, 0, :],
                                  batch_protein_repr_resnet_binary[:, 0, :]),  1)

        logits  = self.multi_classifer(inter_vect3)
        return logits

    



class EmbeddingTransform(nn.Module):

    def __init__(self, input_size, hidden_size, out_size,
                 dropout_p=0.1):
        super(EmbeddingTransform, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size)
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        return hidden


class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self, chem_hidden_size=128,prot_hidden_size=256):
        super(AttentivePooling, self).__init__()
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))

    def forward(self, first, second):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.

        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2, hidden_size)

        Returns:
            (rep_1, attn_1): attention weighted representations and attention scores
            for the first input
            (rep_2, attn_2): attention weighted representations and attention scores
            for the second input
        """

        param = self.param.expand(first.size(0), self.chem_hidden_size,self.prot_hidden_size)

        wm1 = torch.tanh(torch.bmm(second,param.transpose(1,2)))
        wm2 = torch.tanh(torch.bmm(first,param))

        score_m1 = F.softmax(wm1,dim=2)
        score_m2 = F.softmax(wm2,dim=2)

        rep_first = first*score_m1
        rep_second = second*score_m2


        return ((rep_first, score_m1), (rep_second, score_m2))

class ChemicalGraphConv(nn.Module):
    def __init__(self,
                 conv_layer_sizes=[20,20,20,20],
                       output_size=300,
                       degrees=[0,1,2,3,4,5],
                       num_atom_features=num_atom_features(),
                       num_bond_features=num_bond_features()
                 ,use_cuda=None):
        super(ChemicalGraphConv, self).__init__()
        type_map = dict(batch='molecule', node='atom', edge='bond')
        self.model = NeuralFingerprint(
            num_atom_features,
            num_bond_features,
            conv_layer_sizes,
            output_size,
            type_map,
            degrees,use_cuda=use_cuda)

        for param in self.model.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, batch_input, **kwargs):
        batch_embedding = self.model(batch_input)
        return batch_embedding



