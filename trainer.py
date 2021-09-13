import os
import time
import numpy as np
import torch
from sklearn import metrics
from data_tool_box import *
from models import *
from sklearn.metrics import matthews_corrcoef as mcc

#-------------------------
softmax = torch.nn.Softmax(dim=1)
#
def core_batch_prediction(traindf, i, all_config, tokenizer, chem_dict, protein_dict, model,epoch,by_epoch=False,detach=True):
    # ----------------------------------
    #           process input
    # ----------------------------------
    if by_epoch:
        batch_data = traindf[i * all_config['batch_size']:(i + 1) * all_config['batch_size']]
    else:
        batch_data = traindf.sample(all_config['batch_size'])

    batch_chem_graphs, batch_protein_tokenized = get_repr_DTI(batch_data, tokenizer, chem_dict, protein_dict,
                                                              all_config['prot_descriptor'],'contextpred')

    if all_config['use_cuda'] and torch.cuda.is_available():
        batch_protein_tokenized = batch_protein_tokenized.to('cuda')
        # if all_config['chem_option']=='contextpred':
        batch_chem_graphs = batch_chem_graphs.to('cuda')
    # ----------------------------------
    #       get prediction score
    # ----------------------------------
    batch_logits = model(batch_protein_tokenized, batch_chem_graphs,epoch)
    # ----------------------------------
    #            loss
    # ----------------------------------
    batch_labels = torch.LongTensor(batch_data['Activity'].values)
    if all_config['use_cuda'] and torch.cuda.is_available():
        batch_labels = batch_labels.to('cuda')
    if detach == True:
        batch_logits = batch_logits.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

    return batch_logits, batch_labels

def evaluate_multiclass(label, predprobs):
    probs = np.array(predprobs)
    predclass = np.argmax(probs, axis=1)
    # --------------------------by label---
    bothF1 = metrics.f1_score(label, predclass, average=None)
    bothprecision = metrics.precision_score(label, predclass, average=None)
    bothrecall = metrics.recall_score(label, predclass, average=None)
    class0 = [bothF1[0], bothprecision[0], bothrecall[0]]
    class1 = [bothF1[1], bothprecision[1], bothrecall[1]]
    class2 = [bothF1[2], bothprecision[2], bothrecall[2]]
    # -------------------------overall---
    f1 = metrics.cohen_kappa_score(label, predclass)
    auc = metrics.roc_auc_score(label,softmax(torch.tensor(predprobs)).numpy(), multi_class='ovr')
    mcc_score = mcc(label,predclass)
    # fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:, 1], pos_label=1)
    # aupr = metrics.auc(reca, prec)
    # overall ={'f1':f1,'auc':auc,'aupr':aupr}
    overall = [f1,auc,mcc_score]
    return overall,class0,class1,class2

def evaluate3(df, all_config, tokenizer, chem_dict, protein_dict, model, datatype='dev', detach=True):
    collected_logits = []
    collected_labels = []
    epoch = 1000
    for i in range(int(df.shape[0] / all_config['batch_size'])):

        batch_logits, batch_labels = core_batch_prediction(df, i, all_config, tokenizer, chem_dict,
                                                           protein_dict, model,epoch,by_epoch=True,
                                                           detach=True)
        # if len(set(list(batch_labels))) >2:
        collected_logits.append(batch_logits)
        collected_labels.append(batch_labels)
    collected_labels = np.concatenate(collected_labels, axis=0)
    collected_logits = np.concatenate(collected_logits, axis=0)
    overall,class0,class1,class2 = evaluate_multiclass(collected_labels, collected_logits)
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}".format(datatype, overall[0], overall[1], overall[2]))
    return overall,class0,class1,class2



class Trainer_3class():
    def __init__(self, binary_model=None, tokenizer=None,all_config =None,checkpoint_dir=None):
        # ----------------------------------
        #    hyper-parameter/ config
        # ----------------------------------
        self.checkpoint_dir = checkpoint_dir
        self.opt_config= all_config['opt_config']
        self.admin_config=all_config['admin_config']
        self.all_config=all_config
        # ----------------------------------
        #       model
        # ----------------------------------
        self.model = DTI_3_class_V3(all_config=all_config, DTI_binary_pretrained=binary_model)
        if self.all_config['use_cuda'] and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.tokenizer =  tokenizer
        # ----------------------------------
        #       input data
        # ----------------------------------
        self.chem_dict  = pd.read_csv(all_config['cwd']
                                                 + 'data/chemical/'
                                                 + 'ikey2smiles_glass_ango_opo_new_combined.csv')

        self.chem_dict=self.chem_dict.set_index('ikey')['smiles']
        protein_dict_path = 'data/protein/' + 'uni2triplet.pkl'
        self.protein_dict = pd.Series(load_pkl(self.all_config['cwd'] + protein_dict_path))
        print('training by epoch')
    def train(self):
        # ----------------------------------
        #    input data
        # ----------------------------------
        traindf, testdf = load_training_data(
            self.all_config['cwd']
            + 'data/interaction/'
            + self.all_config['exp_mode'],
            self.all_config['debug_ratio'],balanced=False)

        # ----------------------------------
        #    training setup
        # ---------------------------------
        parameters = list(self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.all_config['lr'], weight_decay=self.opt_config['l2'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_target_AUC =-np.inf
        best_epoch = 0
        loss_train = []
        train_performance = {'overall':[],'class0':[],'class1':[],'class2':[]}
        test_performance = {'overall': [], 'class0': [], 'class1': [], 'class2': []}
        print("Data\tF1\tAUC\tmcc")
        # ----------------------------------
        #           training
        # ----------------------------------

        stime=time.time()
        for epoch in range(self.all_config['epochs']):
            print('------------------------epoch:  ',epoch)
            loss_in_epoch=[]
            for i in range(int(traindf.shape[0]/self.all_config['batch_size'])):
                self.model.train()
                batch_logits,batch_labels  = core_batch_prediction(traindf,i,self.all_config,self.tokenizer,
                                                                   self.chem_dict,self.protein_dict,self.model,epoch,
                                                                   detach=False,by_epoch=True)
                loss = loss_fn(batch_logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_in_epoch.append(loss.detach().cpu().numpy())
            loss_train.append(loss_in_epoch)
            # ----------------------------------
            #           evaluation
            # ----------------------------------
            self.model.eval()
            traindf_eval = traindf.sample(frac=0.3) # not evalutate all training data
            testdf_eval=testdf
            print('train eval number:', traindf_eval.shape[0])
            print('test eval number:', testdf_eval.shape[0])
            overall_train,class0_train,class1_train,class2_train=evaluate3(traindf_eval,
                                  self.all_config, self.tokenizer,self.chem_dict,self.protein_dict,self.model,
                                  datatype='train')

            overall_test, class0_test, class1_test, class2_test=evaluate3(testdf_eval,
                                 self.all_config, self.tokenizer,self.chem_dict,self.protein_dict,self.model,
                                 datatype='test')
            train_performance['overall'].append(overall_train)
            train_performance['class0'].append(class0_train)
            train_performance['class1'].append(class1_train)
            train_performance['class2'].append(class2_train)
            test_performance['overall'].append(overall_test)
            test_performance['class0'].append(class0_test)
            test_performance['class1'].append(class1_test)
            test_performance['class2'].append(class2_test)

            np.save(self.checkpoint_dir + 'loss_train.npy', loss_train)
            save_dict_pickle(train_performance,self.checkpoint_dir+'train_performance.pkl')
            save_dict_pickle(test_performance,self.checkpoint_dir+'test_performance.pkl')


            # ----------------------------------
            #           save weights
            # ----------------------------------
            print('time cost of the episode: ', time.time() - stime)
            stime= time.time()
            if test_performance['overall'][-1][1]> best_target_AUC:
                best_target_AUC=  test_performance['overall'][-1][1]
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'model.dat'))
                print('saved at: ', self.checkpoint_dir)
        print("Best test AUC {:.6f} at epoch {}".format(best_target_AUC,best_epoch))
