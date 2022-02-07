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
from fingerprint.features import num_atom_features, num_bond_features
from fingerprint.models import NeuralFingerprint
from resnet import ResnetEncoderModel
#--------------------------

from data_tool_box import *

class DTI_3_class(nn.Module):
    def __init__(self, all_config=None, DTI_binary_pretrained=None):
        super(DTI_3_class,self).__init__()
        self.all_config=all_config
        self.DTI_binary_pretrained = DTI_binary_pretrained
        self.multi_class_interaction_learner = EmbeddingTransform(620,64,3)
        print('620 concat prot/chem/inter vect')
    def forward(self,batch_protein_tokenized,batch_chem_graphs):
        inter_vect = self.DTI_binary_pretrained.embed(batch_protein_tokenized, batch_chem_graphs)
        inter_vect3= self.multi_class_interaction_learner(inter_vect)
        return inter_vect3

class DTI_3_class_no_forgetting(nn.Module):
    def __init__(self, all_config=None, DTI_binary_pretrained=None):
        super(DTI_3_class_no_forgetting,self).__init__()
        self.all_config=all_config
        self.DTI_binary_pretrained = deepcopy(DTI_binary_pretrained)
        self.DTI_binary_pretrained_NO_FORGETTING = deepcopy(DTI_binary_pretrained)
        self.multi_learner = ResnetEncoderModel(1)
        self.multi_classifer = EmbeddingTransform(112, 64, 3)
        print('620 concat prot/chem/inter vect with no-forgetting')
    def forward(self,batch_protein_tokenized,batch_chem_graphs):
        with torch.no_grad():
            inter_vect_no_forgetting = self.DTI_binary_pretrained_NO_FORGETTING.embed(batch_protein_tokenized, batch_chem_graphs)
        inter_vect_updated = self.DTI_binary_pretrained.embed(batch_protein_tokenized, batch_chem_graphs)
        inter_vec = torch.cat((inter_vect_no_forgetting, inter_vect_updated), 1)
        inter_vect3 = self.multi_learner(inter_vec.reshape((self.all_config['batch_size'],2,620)).unsqueeze(1)).reshape(self.all_config['batch_size'],-1)
        logits  = self.multi_classifer(inter_vect3)
        return logits

class DTI_3_class_V3(nn.Module):
    def __init__(self, all_config=None, DTI_binary_pretrained=None):
        super(DTI_3_class_V3,self).__init__()
        self.all_config=all_config
        self.DTI_binary_pretrained = deepcopy(DTI_binary_pretrained)
        # self.multi_learner = ResnetEncoderModel(1)
        print('NEW attentive pooler for multi-class')
        print(all_config['choice'] +'------  with frozen after 50 epochs')
        self.attentive_interaction_pooler = AttentivePooling(300 )
        self.interaction_pooler = EmbeddingTransform(300 + 256, 128, 64, 0.1)
        # self.binary_predictor = EmbeddingTransform(64, 64, 2, 0.2)
        if all_config['choice']=='binary_embed_new_interaction_pipe':
            choice = 64
        elif all_config['choice']=='binary_embed_new_interaction_pipe+binary_embed':
            choice = 620
        elif all_config['choice']=='binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect':
            choice = 684

        if self.all_config['choice']=='all':
            self.multi_learner = ResnetEncoderModel(1)
            self.multi_classifer = EmbeddingTransform(112, 64, 3)
            self.DTI_binary_pretrained_NO_FORGETTING = deepcopy(DTI_binary_pretrained)
        else:
            self.multi_classifer = EmbeddingTransform(choice, 64, 3)


    def forward(self,batch_protein_tokenized,batch_chem_graphs,epoch):
        if epoch>self.all_config['frozen_epoch']:
            with torch.no_grad():
                if self.all_config['choice'] == 'binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect'  or self.all_config['choice'] == 'all':
                    batch_chem_graphs_repr_pooled_binary, batch_protein_repr_resnet_binary, interaction_vector_binary = self.DTI_binary_pretrained.embed3(
                        batch_protein_tokenized, batch_chem_graphs)
                else:
                    batch_chem_graphs_repr_pooled_binary, batch_protein_repr_resnet_binary = self.DTI_binary_pretrained.embed2(
                        batch_protein_tokenized, batch_chem_graphs)
                # elif all_config['choice'] == 'binary_embed_new_interaction_pipe+binary_embed' or all_config['choice']=='binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect':
                if self.all_config['choice'] == 'all':
                    with torch.no_grad():
                        batch_chem_graphs_repr_pooled_binary_no_forgetting, batch_protein_repr_resnet_binary_no_forgetting,interaction_vector_binary_no_forgetting = self.DTI_binary_pretrained_NO_FORGETTING.embed3(
                            batch_protein_tokenized, batch_chem_graphs)
        else:
            if self.all_config['choice'] == 'binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect' or \
                    self.all_config['choice'] == 'all':
                batch_chem_graphs_repr_pooled_binary, batch_protein_repr_resnet_binary, interaction_vector_binary = self.DTI_binary_pretrained.embed3(
                    batch_protein_tokenized, batch_chem_graphs)
            else:
                batch_chem_graphs_repr_pooled_binary, batch_protein_repr_resnet_binary = self.DTI_binary_pretrained.embed2(
                    batch_protein_tokenized, batch_chem_graphs)
            # elif all_config['choice'] == 'binary_embed_new_interaction_pipe+binary_embed' or all_config['choice']=='binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect':
            if self.all_config['choice'] == 'all':
                with torch.no_grad():
                    batch_chem_graphs_repr_pooled_binary_no_forgetting, batch_protein_repr_resnet_binary_no_forgetting, interaction_vector_binary_no_forgetting = self.DTI_binary_pretrained_NO_FORGETTING.embed3(
                        batch_protein_tokenized, batch_chem_graphs)
        ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(batch_chem_graphs_repr_pooled_binary,    batch_protein_repr_resnet_binary) # same as input dimension
        inter_vect3 = self.interaction_pooler(  torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        if self.all_config['choice'] == 'binary_embed_new_interaction_pipe+binary_embed':
            inter_vect3 = torch.cat((inter_vect3, batch_chem_graphs_repr_pooled_binary[:,0,:], batch_protein_repr_resnet_binary[:,0,:]), 1)
        elif self.all_config['choice'] == 'binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect':
            inter_vect3 = torch.cat( (inter_vect3,
                                      interaction_vector_binary,
                                      batch_chem_graphs_repr_pooled_binary[:, 0, :],
                                      batch_protein_repr_resnet_binary[:, 0, :]),  1)
        elif self.all_config['choice'] == 'all':
            inter_vect3 = torch.cat( (inter_vect3,
                                      interaction_vector_binary,
                                      batch_chem_graphs_repr_pooled_binary[:, 0, :],
                                      batch_protein_repr_resnet_binary[:, 0, :],
                                      batch_chem_graphs_repr_pooled_binary_no_forgetting[:, 0, :],
                                      batch_protein_repr_resnet_binary_no_forgetting[:, 0, :],
                                      interaction_vector_binary_no_forgetting),  1)
            inter_vect3 = self.multi_learner( inter_vect3.reshape((self.all_config['batch_size'], 2, 652)).unsqueeze(1)).reshape(self.all_config['batch_size'], -1)
            # logits = self.multi_classifer(inter_vect3)
        # else:
        logits  = self.multi_classifer(inter_vect3)
        return logits

    
class DTI_model(nn.Module):
    def __init__(self, all_config=None,
                 contextpred_config = {
                            'num_layer':5,
                            'emb_dim':300,
                            'JK':'last',
                            'drop_ratio':0.5,
                            'gnn_type':'gin'
                 },
                 model=None):
        super(DTI_model, self).__init__()
        # -------------------------------------------
        #         hyper-parameter
        # -------------------------------------------
        self.use_cuda = all_config['use_cuda']
        self.contextpred_config= contextpred_config
        self.all_config = all_config

        # -------------------------------------------
        #         model components
        # -------------------------------------------

       #          chemical decriptor
        if all_config['chem_option']=='contextpred':
            self.ligandEmbedding = GNN(num_layer=contextpred_config['num_layer'],
                                       emb_dim=contextpred_config['emb_dim'],
                                       JK=contextpred_config['JK'],
                                       drop_ratio=contextpred_config['drop_ratio'],
                                       gnn_type=contextpred_config['gnn_type'])

        else:
            self.ligandEmbedding = ChemicalGraphConv(use_cuda=self.use_cuda)
        #          protein decriptor
        proteinEmbedding =model
        self.proteinEmbedding  = proteinEmbedding
        if all_config['prot_descriptor']=='DISAE':

            prot_embed_dim=256 # out of resnet

        if all_config['prot_frozen']=='partial':
            prot_embed_dim = 256
            ct = 0
            for m in self.proteinEmbedding.modules():
                ct += 1
                if ct in all_config['DISAE']['frozen_list']:
                    print('frozen module ', ct)
                    for param in m.parameters():
                        param.requires_grad = False
                else:
                    for param in m.parameters():
                        param.requires_grad = True

        self.resnet = ResnetEncoderModel(1)

        #        interaction
        self.attentive_interaction_pooler = AttentivePooling(contextpred_config['emb_dim'], )
        self.interaction_pooler  = EmbeddingTransform(contextpred_config['emb_dim'] +prot_embed_dim,128,64,0.1)
        self.binary_predictor = EmbeddingTransform(64,64,2,0.2)

        if self.use_cuda and torch.cuda.is_available():
            self.attentive_interaction_pooler = self.attentive_interaction_pooler.to('cuda')
            self.interaction_pooler = self.interaction_pooler.to('cuda')
            self.binary_predictor = self.binary_predictor.to('cuda')
            self.ligandEmbedding = self.ligandEmbedding.to('cuda')
            self.proteinEmbedding = self.proteinEmbedding.to('cuda')


        
    def forward(self, batch_protein_tokenized,batch_chem_graphs, epoch,**kwargs):
        # ---------------protein embedding ready -------------
        if self.all_config['prot_descriptor']=='DISAE':
            if self.all_config['frozen'] == 'whole':
                with torch.no_grad():
                    batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
            else:
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0] #(batch_size,len=212,emb=312,)

            batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(self.all_config['batch_size'],1,-1)#(batch_size,1,256)

        # ---------------ligand embedding ready -------------
        if self.all_config['chem_option']=='contextpred':
            node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                       batch_chem_graphs.edge_attr)
            batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation, batch_chem_graphs.batch)#(batch_size,len,300)
            batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1) #(batch_size,1,300)
        else:
            batch_chem_graphs_repr_pooled = self.ligandEmbedding(batch_chem_graphs)#(batch_size,1,300)
        # ---------------interaction embedding ready -------------
        ((chem_vector,chem_score),(prot_vector,prot_score)) = self.attentive_interaction_pooler(batch_chem_graphs_repr_pooled,
                                                                        batch_protein_repr_resnet) #same as input dimension


        interaction_vector = self.interaction_pooler(torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))#(batch_size,64)
        logits =self.binary_predictor(interaction_vector) #(batch_size,2)
        return logits


class DTI_model_pretrained(nn.Module):
    def __init__(self, all_config=None,
                 contextpred_config={
                     'num_layer': 5,
                     'emb_dim': 300,
                     'JK': 'last',
                     'drop_ratio': 0.5,
                     'gnn_type': 'gin'
                 },
                 model=None):
        super(DTI_model_pretrained, self).__init__()
        # -------------------------------------------
        #         hyper-parameter
        # -------------------------------------------
        self.use_cuda = all_config['use_cuda']
        self.contextpred_config = contextpred_config
        self.all_config = all_config

        # -------------------------------------------
        #         model components
        # -------------------------------------------

        #          chemical decriptor
        if all_config['chem_option'] == 'contextpred':
            self.ligandEmbedding = GNN(num_layer=contextpred_config['num_layer'],
                                       emb_dim=contextpred_config['emb_dim'],
                                       JK=contextpred_config['JK'],
                                       drop_ratio=contextpred_config['drop_ratio'],
                                       gnn_type=contextpred_config['gnn_type'])

        else:
            self.ligandEmbedding = ChemicalGraphConv(use_cuda=self.use_cuda)
        #          protein decriptor
        proteinEmbedding = model
        self.proteinEmbedding = proteinEmbedding
        if all_config['prot_descriptor'] == 'DISAE':
            prot_embed_dim = 256  # out of resnet

        if all_config['prot_frozen'] == 'partial':
            prot_embed_dim = 256
            ct = 0
            for m in self.proteinEmbedding.modules():
                ct += 1
                if ct in all_config['DISAE']['frozen_list']:
                    print('frozen module ', ct)
                    for param in m.parameters():
                        param.requires_grad = False
                else:
                    for param in m.parameters():
                        param.requires_grad = True

        self.resnet = ResnetEncoderModel(1)

        #        interaction
        self.attentive_interaction_pooler = AttentivePooling(contextpred_config['emb_dim'], )
        self.interaction_pooler = EmbeddingTransform(contextpred_config['emb_dim'] + prot_embed_dim, 128, 64, 0.1)
        self.binary_predictor = EmbeddingTransform(64, 64, 2, 0.2)

        if self.use_cuda and torch.cuda.is_available():
            self.attentive_interaction_pooler = self.attentive_interaction_pooler.to('cuda')
            self.interaction_pooler = self.interaction_pooler.to('cuda')
            self.binary_predictor = self.binary_predictor.to('cuda')
            self.ligandEmbedding = self.ligandEmbedding.to('cuda')
            self.proteinEmbedding = self.proteinEmbedding.to('cuda')

    def forward(self, batch_protein_tokenized, batch_chem_graphs, **kwargs):
        # ---------------protein embedding ready -------------
        # if self.all_config['prot_descriptor'] == 'DISAE':
        if self.all_config['binary_frozen'].split('-')[0] == 'whole':
            # print('frozen binary protein descriptor')
            with torch.no_grad():
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
        else:
            batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]  # (batch_size,len=212,emb=312,)

        batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(
            self.all_config['batch_size'], 1, -1)  # (batch_size,1,256)

        # ---------------ligand embedding ready -------------
        if self.all_config['chem_option'] == 'contextpred':
            if self.all_config['binary_frozen'].split('-')[1]=='whole':
                # print('frozen binary chemical descriptor')
                with torch.no_grad():
                    node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                       batch_chem_graphs.edge_attr)
            else:
                node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                           batch_chem_graphs.edge_attr)
            batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                       batch_chem_graphs.batch)  # (batch_size,len,300)
            batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1)  # (batch_size,1,300)
        # else:
        #     batch_chem_graphs_repr_pooled = self.ligandEmbedding(batch_chem_graphs)  # (batch_size,1,300)
        # ---------------interaction embedding ready -------------
        if self.all_config['binary_frozen'].split('-')[2]=='whole':
            # print('frozen binary attentive pooler and interaction pooler')
            with torch.no_grad():
                ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
                    batch_chem_graphs_repr_pooled,
                    batch_protein_repr_resnet)  # same as input dimension

                interaction_vector = self.interaction_pooler(
                    torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        else:
            ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
                batch_chem_graphs_repr_pooled,
                batch_protein_repr_resnet)  # same as input dimension

            interaction_vector = self.interaction_pooler(
                torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        # logits = self.binary_predictor(interaction_vector)  # (batch_size,2)
        return interaction_vector
    def embed3(self,batch_protein_tokenized, batch_chem_graphs, **kwargs):
        # ---------------protein embedding ready -------------
        # if self.all_config['prot_descriptor'] == 'DISAE':
        if self.all_config['binary_frozen'].split('-')[0] == 'whole':
            # print('frozen binary protein descriptor')
            with torch.no_grad():
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
        else:
            batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]  # (batch_size,len=212,emb=312,)

        batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(
            self.all_config['batch_size'], 1, -1)  # (batch_size,1,256)

        # ---------------ligand embedding ready -------------
        if self.all_config['chem_option'] == 'contextpred':
            if self.all_config['binary_frozen'].split('-')[1] == 'whole':
                # print('frozen binary chemical descriptor')
                with torch.no_grad():
                    node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                               batch_chem_graphs.edge_attr)
            else:
                node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                           batch_chem_graphs.edge_attr)
            batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                       batch_chem_graphs.batch)  # (batch_size,len,300)
            batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1)  # (batch_size,1,300)
        # else:
        #     batch_chem_graphs_repr_pooled = self.ligandEmbedding(batch_chem_graphs)  # (batch_size,1,300)
        # ---------------interaction embedding ready -------------
        if self.all_config['binary_frozen'].split('-')[2] == 'whole':
            # print('frozen binary attentive pooler and interaction pooler')
            with torch.no_grad():
                ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
                    batch_chem_graphs_repr_pooled,
                    batch_protein_repr_resnet)  # same as input dimension

                interaction_vector = self.interaction_pooler(
                    torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        else:
            ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
                batch_chem_graphs_repr_pooled,
                batch_protein_repr_resnet)  # same as input dimension

            interaction_vector = self.interaction_pooler(
                torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        # logits = self.binary_predictor(interaction_vector)  # (batch_size,2)
        # vector = torch.cat((interaction_vector,batch_chem_graphs_repr_pooled[:,0,:], batch_protein_repr_resnet[:,0,:]), 1)
        return batch_chem_graphs_repr_pooled, batch_protein_repr_resnet,interaction_vector
    def embed2(self,batch_protein_tokenized, batch_chem_graphs, **kwargs):
        # ---------------protein embedding ready -------------
        # if self.all_config['prot_descriptor'] == 'DISAE':
        if self.all_config['binary_frozen'].split('-')[0] == 'whole':
            # print('frozen binary protein descriptor')
            with torch.no_grad():
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
        else:
            batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]  # (batch_size,len=212,emb=312,)

        batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(
            self.all_config['batch_size'], 1, -1)  # (batch_size,1,256)

        # ---------------ligand embedding ready -------------
        if self.all_config['chem_option'] == 'contextpred':
            if self.all_config['binary_frozen'].split('-')[1] == 'whole':
                # print('frozen binary chemical descriptor')
                with torch.no_grad():
                    node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                               batch_chem_graphs.edge_attr)
            else:
                node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                           batch_chem_graphs.edge_attr)
            batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation,
                                                                       batch_chem_graphs.batch)  # (batch_size,len,300)
            batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1)  # (batch_size,1,300)
        # else:
        #     batch_chem_graphs_repr_pooled = self.ligandEmbedding(batch_chem_graphs)  # (batch_size,1,300)
        # ---------------interaction embedding ready -------------
        # if self.all_config['binary_frozen'].split('-')[2] == 'whole':
        #     # print('frozen binary attentive pooler and interaction pooler')
        #     with torch.no_grad():
        #         ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
        #             batch_chem_graphs_repr_pooled,
        #             batch_protein_repr_resnet)  # same as input dimension
        #
        #         interaction_vector = self.interaction_pooler(
        #             torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        # else:
        #     ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
        #         batch_chem_graphs_repr_pooled,
        #         batch_protein_repr_resnet)  # same as input dimension
        #
        #     interaction_vector = self.interaction_pooler(
        #         torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        # logits = self.binary_predictor(interaction_vector)  # (batch_size,2)
        # vector = torch.cat((batch_chem_graphs_repr_pooled[:,0,:], batch_protein_repr_resnet[:,0,:]), 1)
        return batch_chem_graphs_repr_pooled, batch_protein_repr_resnet

class EmbeddingTransform2(nn.Module):

    def __init__(self, input_size, hidden_size, out_size,
                 dropout_p=0.1):
        super(EmbeddingTransform2, self).__init__()
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

class AttentivePooling2(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self,embedding_length = 300):
        super(AttentivePooling2, self).__init__()
        self.embedding_length = embedding_length
        self.U = nn.Parameter(torch.zeros(self.embedding_length, self.embedding_length))

    def forward(self, protein, ligand):
        """ Calculate attentive pooling attention weighted representation and

        """

        U= self.U.expand(protein.size(0), self.embedding_length,self.embedding_length)
        Q = protein
        A = ligand
        G = torch.tanh(torch.bmm(torch.bmm(Q, U), A.transpose(1,2)))
        g_q = G.max(axis=2).values
        g_a = G.max(axis=1).values

        def get_attention_score(g_q,Q):
            g_q_masked = g_q.masked_fill(g_q == 0, -1e9)
            sigma_q = F.softmax(g_q_masked)
            prot_repr = Q * sigma_q[:, :, None]
            prot_vec = prot_repr.sum(1)
            return sigma_q,prot_vec

        sigma_q, prot_vec = get_attention_score(g_q,Q)
        sigma_a, chem_vec = get_attention_score(g_a,A)

        return sigma_q, prot_vec, sigma_a, chem_vec
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



