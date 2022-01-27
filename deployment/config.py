"""
    Description: functions for text preprocessing
    Author: Jimmy L.
    Date: Spring 2022
"""
import torch
import config
from transformer import BertUncased, predict
from transformers import BertTokenizer
from Label2Id import Label2Id


label_tknz = Label2Id()
label_tknz.load_dict("deployment/model/label_dict/label_dict.pkl")


input_tknz = BertTokenizer.from_pretrained("deployment/model/input_tknz")

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert = BertUncased()
bert.build_model(len(label_tknz))
bert.load_weights("deployment/model/weights/checkpoint.pth", device=device)


# hyperparameters for the A.I. model
max_len = 200