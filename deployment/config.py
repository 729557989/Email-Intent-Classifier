import torch
import config
from transformer import BertUncased, predict
from transformers import BertTokenizer
from Label2Id import Label2Id


label_tknz = Label2Id()
label_tknz.load_dict("model/label_dict/label_dict.pkl")


input_tknz = BertTokenizer.from_pretrained("model/input_tknz")


bert = BertUncased()
bert.build_model(len(label_tknz))
bert.load_weights("model\weights\checkpoint.pth")


# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters for the A.I. model
max_len = 200