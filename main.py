import os
import time

import torch
from torch.utils.data import DataLoader

from train_test import train_model, test_model
from model import deepGBLUP
#from dataset import load_dataset
from dataset_edit import load_dataset

################ CONFIG ####################
# data path
raw_path = 'data_sim/output_geno_test.raw' # path of raw file
phen_path = 'data_sim/output_pheno_test.phen' # path of phenotype file
bim_path = None # optional: path of bim file to save SNP effects. If you don't have bim file just type None 

# train cofig
lr =  1e-5 # list of candidate learning rate
epoch = 4 # max value of candidate epoch
batch_size = 16 # train batch size

device = 'cpu' # type 'cpu' if you use cpu device, or type 'cuda' if you use gpu device.
h2 = 0.1

# save config
save_path = 'my_outputs_test' # path to save results

##############################################

####################################################
##                      Caution                   ## 
##  Users unfamiliar with python and pytorch      ##
##  should not modify the code below.             ##
####################################################
os.makedirs(save_path, exist_ok=True)
s_time = time.time()

# Load Dataset
train_dataset, test_dataset, num_snp, ymean = load_dataset(raw_path, phen_path, h2, device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = deepGBLUP(ymean, num_snp).to(device)

# Train model
train_time  = train_model(
    model, train_dataloader, test_dataloader, 
    save_path, device, lr, epoch, h2
    )

# test model
test_time = test_model(
        model, 
        test_dataloader,
        device, save_path
        )

# Save hyperparameters
with open(os.path.join(save_path,'setting.txt'),'w') as save:
    save.write(f'Path of raw file: {raw_path}\n')
    save.write(f'Path of phenotype file: {phen_path}\n')
    save.write('-'*50+'\n')
    save.write(f'learning rate: {lr}\n')
    save.write(f'epoch: {epoch}\n')
    save.write(f'batch_size: {batch_size}\n')
    save.write(f'Device: {device}\n')
    save.write('-'*50+'\n')
    save.write(f'H2: {h2}\n')
    save.write('train time\t'+str(train_time)+'\n')
    save.write('test time\t'+str(test_time)+'\n')
    save.write('running time\t'+str(time.time() - s_time)+'\n') # add total runtime
    
