import torch

def generate_default_config():
    configs = {}
    
    # Device
    configs['use_gpu'] = torch.cuda.is_available()
    configs['use_multi_gpu'] = configs['use_gpu'] and torch.cuda.device_count() > 1
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() and configs['use_gpu'] else 'cpu')
    
    # Dataset
    configs['dataset'] = None
    
    # Training parameters
    # configs['dtype'] = torch.float
    configs['lr'] = 1e-1
    configs['weight_decay'] = 1e-4
    configs['batch_size'] = 512
    # configs['num_met_layer'] = 1
    # configs['start_epoch'] = 0
    # configs['max_epoch'] = 200
    # configs['evaluate'] = False
    # configs['pre_sigmoid_bias'] = 0.5
    
    # # Training information display and log
    # configs['display'] = True
    # configs['display_freq'] = 10
    # configs['save_checkpoint_path'] = 'checkpoint'
    # configs['exp'] = 'exp'
    
    # Reproducibility
    configs['rand_seed'] = 0
    
    return configs

def Adult_configs(configs):
    configs['lr'] = 0.01

def BeLaE_configs(configs):
    configs['lr'] = 0.01

def CoIL2000_configs(configs):
    configs['lr'] = 0.1

def Default_configs(configs):
    configs['lr'] = 0.01

def Enb_configs(configs):
    configs['lr'] = 0.1

def Flare1_configs(configs):
    configs['lr'] = 0.01
    
def Flickr_configs(configs):
    configs['lr'] = 0.1

def Jura_configs(configs):
    configs['lr'] = 0.1

def Oes10_configs(configs):
    configs['lr'] = 0.1

def Oes97_configs(configs):
    configs['lr'] = 0.1

def Song_configs(configs):
    configs['lr'] = 0.01

def Thyroid_configs(configs):
    configs['lr'] = 0.01

def TIC2000_configs(configs):
    configs['lr'] = 0.01

def Voice_configs(configs):
    configs['lr'] = 0.01

