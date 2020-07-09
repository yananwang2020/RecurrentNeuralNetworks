from RNNVanilla import RNN_Vanilla
from pathlib import Path
import numpy as np

file_name = "The_Old_Man_and_the_Sea-Ernest_Hemingway"
txt_file_path = Path(f'./Lib/{file_name}.txt')
param_file_path = Path(f'./Param/{file_name}.pkl')

txt = ''
txt_lines = []
with txt_file_path.open('r') as fr:
    txt = fr.read()
    #raw_lines = txt.split('\n')
    #txt_lines[:] = (paragraph+'\n' for paragraph in raw_lines if paragraph != '')

# load txt from the file
chars = list(set(txt))
txt_size, input_size = len(txt), len(chars)
print(f'Whole txt has {txt_size} characters, {input_size} uniques')

# unit dictionary
unit_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_unit = {i:ch for i,ch in enumerate(chars)}

hidden_size = 512
bunch_size = 20
learn_rate = 0.01

h_pre = np.zeros((hidden_size, 1))
rnn_model = RNN_Vanilla(hidden_size, input_size, learn_rate)
rnn_model.LoadParam(param_file_path)

crt_iter = 0
bunch_idx = 0
bunch_end = bunch_idx + bunch_size + 1
train_iter = 1000
sample_iter = (250, 500, 750, 1000)
while crt_iter < train_iter:
    crt_iter+=1

    # select train bunch
    bunch_idx = min(bunch_idx, txt_size-1)
    bunch_end = min(bunch_end, txt_size-1)

    bunch_str = txt[bunch_idx : bunch_end]
    bunch_data = [unit_to_idx[unit] for unit in bunch_str]

    if(bunch_end >= txt_size-1):
        bunch_idx = 0
    else:
        bunch_idx += bunch_size
    bunch_end = bunch_idx + bunch_size + 1
    
    # train
    loss, h_pre = rnn_model.Epoch(bunch_data, h_pre)

    # sample
    if(crt_iter in sample_iter):
        print(f'Iter: {crt_iter}, Loss: {loss}')    

rnn_model.SaveParam(param_file_path)


