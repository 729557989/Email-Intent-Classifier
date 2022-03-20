"""
    Description: map string labels into integer ids
    Author: Jimmy L. @ AI - Camp
    Date: Spring 2022
"""
import json

#TODO: idea, every time when use it, just previde it's self.dict instead of saving the tokenizer

class Label2Id:
    """
    Purpose: A class that serves similarly like the Tensorflow.Keras tokenizer
    """
    def __init__(self, label_dict=None):
        # The dictionay that stores via: label, id
        self.dict = label_dict if label_dict != None else {}

        # A dictionary that records how many instances of labels
        self.count = {}

        # The dictionay that stores via: id, label
        self.reverse_dict = {}

    def fit(self, labels):
        # Takes in a list of labels to update the self.count
        for word in labels:
            self.count[word] = self.count.get(word, 0) + 1
        temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)
        self.count = dict(temp)

    def build_vocab(self, labels):
        # Takes in a list of labels to update the self.dict
        for label in labels:
            if label not in self.dict.keys():
                self.dict[label] = len(self.dict)
        
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
    
    def encoder(self, label):
        # Converts a string label to an integer id
        return self.dict.get(label)
    
    def decoder(self, id):
        # Converts an integer id to a string label
        return self.reverse_dict.get(id)
    
    def __len__(self):
        # Return the vocab size stored in the self.dict
        return len(self.dict)
    
    def save_dict(self, path):
        # Save self.dict at location path
        with open(path, 'w') as file_object:  #open the file in write mode
            json.dump(self.dict, file_object)
            
        print("Successfully Saved label dict!")

    def load_dict(self, path):
        # Load the saved dictionary to self.dict
        
        file = open(path, 'rb')
        self.dict = json.load(file)
        
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        print("Successfully Loaded label dict!")


#NOTE: pickle save and load methods below
"""
To SAVE:
pickle.dump(object, open(path, "wb"))

To LOAD:
pickle.load(open(path,'rb'))
"""