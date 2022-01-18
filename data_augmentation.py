"""
    Description: config file for hyperparameters
    Author: Jimmy L.
    Date: Spring 2022

NOTE: Installed Parrot with: pip install git+https://github.com/PrithivirajDamodaran/Parrot.git
      library needed for Parrot: pip install protobuf
"""


from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(use_gpu=False) # model_tag="prithivida/parrot_paraphraser_on_T5", 

phrases = ["I was wondering if or when we would be able to decide the week we begin our camp? Your website lists three different weeks and I had a specific week in mind, so I was just wondering if we choose or not. Thank you"]

for phrase in phrases:
  print("-"*100)
  print("Input_phrase: ", phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase=phrase)
  for para_phrase in para_phrases:
    print(para_phrase)