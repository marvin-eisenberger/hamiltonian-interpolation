import torch
import pathlib

path_curr = str(pathlib.Path(__file__).parent.absolute())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
data_folder_faust = path_curr + "/../data/"
data_folder_out = path_curr + "/../data/"

