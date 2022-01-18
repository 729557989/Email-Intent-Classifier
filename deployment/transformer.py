"""
    Description: construct the NLP Transformer model here
    Author: Jimmy L.
    Date: Spring 2022
"""
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from text_preprocessing import clean_texts
import config
import torch
import numpy as np
# disable HuggingFace Warning
import logging
logging.disable(logging.WARNING)
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')



class BertUncased():
    """
    A Class for A.I model, optimizer, and scheduler storage
    """
    def __init__(self):
        self.optimizer = None
        self.lrScheduler = None
        self.model = None

    def build_model(self, num_labels):
        # Class method to initialize the A.I. model -> self.model
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",          # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels,      # The number of output labels
            output_attentions = False,    # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    
    def load_weights(self, filepath, predict_mode=True):
        # Class method to load pretrained model weights into the A.I. model -> self.model
        # NOTE: this loading method is for making predictions 
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if predict_mode == True:
            self.model.eval()

    def get_model_parameters(self):
        # Prints our the A.I. model -> self.model's layer architecture
        assert self.model != None
        print("<" + "-"*80 + ">")
        params = list(self.model.named_parameters())
        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print("<" + "-"*80 + ">")

    def build_optimizer(self, learning_rate):
        # Class method to initialize the optimizer -> self.optimizer
        assert self.model != None
        self.optimizer = AdamW(self.model.parameters(),
            lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps = 1e-8          # args.adam_epsilon  - default is 1e-8.
        )

    def build_LRscheduler(self, train_dataloader, epochs):
        # Class method to initialize the scheduler -> self.scheduler
        assert self.optimizer != None
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)



def predict(input, model, input_tknz, label_tknz, device=None):
    """
    Purpose: Utilize the A.I. model to make predictions

    Params:  1. input (1d list of string sentences):
                The preprocessed string sentence the model is trying to make prediction of.

             2. model (BertForSequenceClassification):
                The model to make predictions with.

             3. input_tknz (transformers.models.bert.tokenization_bert.BertTokenizer):
                    The input tokenizer that transform string inputs into a list of pytorch float

             4. label_tknz (Label2Id object):
                Label tokenizer that converts string labels into integers.

             5. device (torch.device):
                What device to use for A.I. training, generally 'cpu' or 'cuda'

    Returns: A list of predictions containing string labels
    """
    
    # preprocess the input with function clean_texts from file text_preprocessing.py
    preprocessed_input = clean_texts(input)

    # set model to evaluation mode
    model.eval()

    # set model to utilize device
    model.to(device if device != None else 'cpu')
    predictions = []
    for sent in preprocessed_input:

        # Detach any tensors with gradients that are currently attached to the computational graph.
        with torch.no_grad():
            input_seqs = []
            attention_masks = []
            """
            "encode_plus" will:
            (1) Tokenize the sentence.
            (2) Prepend the `[CLS]` token to the start.
            (3) Append the `[SEP]` token to the end.
            (4) Map tokens to their IDs.
            (5) Pad or truncate the sentence to `max_length`
            (6) Create attention masks for [PAD] tokens
            """
            encoded_dict = input_tknz.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = config.max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
            
            # append outputs from input tokenizer
            input_seqs.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            input_seqs = torch.cat(input_seqs, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            # cast inputs to utilize device
            input_seqs = input_seqs.to(device if device != None else 'cpu')
            attention_masks = attention_masks.to(device if device != None else 'cpu')
            output = model(input_seqs, 
                     token_type_ids=None, 
                     attention_mask=attention_masks,
                     return_dict=True)
            
            # cast outputs to numpy
            output_np = output[0].to('cpu').numpy()
            str_output = label_tknz.decoder(np.argmax(output_np[0]))
            predictions.append(str_output)
    
    # get list of predictions
    return predictions