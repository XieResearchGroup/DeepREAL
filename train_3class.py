"""
DeepFunLib based on DISAE
protein descriptor:  DISAE-plus
chemical descriptor: contextPred or NeuralFingerPrint
data: GLASS and targets angonist/antagonist
"""
# ------------- admin
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import numpy as np
import torch

# -------------  my work
from models import *
from trainer import *
from utils import  *
from data_tool_box import *

#-------------------------------------------
#      set hyperparameters
#-------------------------------------------

parser = argparse.ArgumentParser("DeepFunLib")
# ---------- args for admin
parser.add_argument('--cwd', type=str, default='')
parser.add_argument('--debug_ratio', type=float, default=1.0)
# parser.add_argument('--few_training_data', type=float, default=1.0)
parser.add_argument('--exp_mode', default='the_3way_chem_split/',help='Path to the train/dev/test dataset.')
parser.add_argument('--pretrained_binary_path', default='exp29-08-2021-01-56-03/')
#---------- args for protein descriptor
parser.add_argument('--prot_descriptor', type=str, default='DISAE',help='choose from [DISAE, TAPE,ESM ]')
parser.add_argument('--DISAE_raw', type=str2bool, default=False)
parser.add_argument('--prot_frozen', type=str, default='none',help='choose from {whole, none,partial}')
parser.add_argument('--frozen', type=str, default='none',help='choose from {whole, none,partial}')
parser.add_argument('--binary_frozen', type=str, default='none-none-none',help='{none, whole-whole-whole}')
#---------- args for ContextPred
# parser.add_argument('--chem_option',type=str,default='contextpred',help='chose from {neuralfp, contextpred}')
parser.add_argument('--pretrained_onBinary',type=str2bool, nargs='?',const=True, default=True)
####---------- args for model training and optimization
# parser.add_argument('--balanced_test',type=str2bool, nargs='?',const=True, default=F暗蓝色)
# parser.add_argument('--choice',type=str,default='binary_embed_new_interaction_pipe+binary_embed+binary_inter_vect',  help='chose from {+binary_embed,++binary_inter_vect}')
# parser.add_argument('--warmed_up',type=str,default='',help='pretrained on some other 3 class prediction datasets as a warm up,format:exp**')
# parser.add_argument('--global_step', default=30, type=int, help='Number of training epoches ')
parser.add_argument('--epochs', default=30, type=int, help='Number of training epoches ')
# parser.add_argument('--frozen_epoch', default=50, type=int, help='Number of training epoches ')
# parser.add_argument('--eval_at', default=10, type=int, help='')
parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
parser.add_argument('--lr', type=float, default=2e-5, help="Initial learning rate")
#----------

opt = parser.parse_args()
# -------------------------------------------
#         set admin
# -------------------------------------------
all_config = load_json(opt.cwd + 'DTI_config.json')
checkpoint_dir = set_up_exp_folder(opt.cwd,'exp_logs_3class/')
np.random.seed(7)
seed = all_config['opt_config']['random_seed']
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.set_num_threads(all_config['opt_config']['num_threads'])

all_config.update(vars(opt))

config_file = checkpoint_dir + 'config.json'
save_json(all_config,
          config_file)  # combine config files with the most frequently tuned ones and save again just in case
if opt.use_cuda == False:
    print('not using GPU')

# ------------- protein descriptor
if all_config['prot_descriptor']=='DISAE':
    print('using DISAE+')
    from transformers import BertTokenizer
    from transformers.configuration_albert import AlbertConfig
    from transformers.modeling_albert import AlbertForMaskedLM
    from transformers.modeling_albert import load_tf_weights_in_albert

#-------------------------------------------
#         main
#-------------------------------------------
if __name__ == '__main__':
    print(all_config['exp_mode'])
    # -------------------------------------------
    #      set up DTI models
    # -------------------------------------------
    # Load protein descriptor
    if all_config['prot_descriptor'] == 'DISAE':
        albertconfig = AlbertConfig.from_pretrained(all_config['cwd']+all_config['DISAE']['albertconfig'])
        m = AlbertForMaskedLM(config=albertconfig)
        if all_config['DISAE_raw']==False:
            m = load_tf_weights_in_albert(m, albertconfig,
                                                    all_config['cwd']+all_config['DISAE']['albert_pretrained_checkpoint'])
        else:
            print('DISAE raw')
        prot_descriptor = m.albert
        prot_tokenizer = BertTokenizer.from_pretrained(all_config['cwd']+all_config['DISAE']['albertvocab'])


    DTI_model_pretrained_instance = DTI_model_pretrained( all_config = all_config ,model= prot_descriptor)
    if all_config['pretrained_onBinary']==True:
        print('pretrained on binary GLASS+')
        DTI_pretrained_path = all_config['cwd']+'exp_logs_binary_pretraining/'+all_config['pretrained_binary_path']+'model.dat'
        DTI_model_pretrained_instance.load_state_dict(torch.load(DTI_pretrained_path))
    else:
        print('3 class classification training from scratch')
    # -------------------------------------------
    #      set up trainer and evaluator
    # -------------------------------------------


    trainer = Trainer_3class(binary_model=DTI_model_pretrained_instance, tokenizer=prot_tokenizer,
                      all_config = all_config,checkpoint_dir=checkpoint_dir)


    # -------------------------------------------
    #      training and evaluating
    # -------------------------------------------

    trainer.train()
    print('Finished training! Experiment log at: ', checkpoint_dir)

