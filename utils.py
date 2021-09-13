import os
from datetime import datetime
from data_tool_box import *
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def set_up_exp_folder(cwd,name):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = cwd + name
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir

def set_config_to_json(path='D:/Projects_on_going/distance-matrix/'):
    admin_config = {
                    'cwd' : 'D:/Projects_on_going/distance-matrix/',
                    'chemble_path' :'data/ChEMBLE26/'
                    }
    chem_config  = {'chem_dropout' : 0.5,
                    'ckpt_chem' : 'chem_side_pretraining/pretrained_14.pt',
                    'dim_features':6,
                    'dim_target':312,
                    'n_in':312,
                    'n_out':740 ,#help='Pubchem fingerprint dimension',
                    'hidden_units':[64, 64, 64, 64],
                    'train_eps':True,
                    'aggregation': 'sum',
                            }

    interaction_config = {
                        'ap_dropout': 0.1,
                        'ap_feature_size':64
                        }
    opt_config = {
                    'random_seed':705,
                    'max_eval_steps': 1000,
                    'optimizer':'adam',
                    'scheduler':'cosineannealing',
                    'l2':1e-4,
                    'num_threads':8
                    }
    config =  {'admin_config': admin_config,
                'chem_config' : chem_config,
                'interaction_config' : interaction_config,
                'opt_config' : opt_config}
    save_json(config,path+'DTI_config_ESM_ContextPred.json')



