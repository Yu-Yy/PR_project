import csv
import numpy as np
import pickle
import math

text_path = "../shopee-product-matching/train.csv"  
class_spe = 'class_specific.pkl'
creating = False
if creating:
    dic_Text = {}
    with open(text_path, 'r') as f:
        reader = csv.reader(f)
        mask = 0
        for row in reader:
            if mask == 0: # gap the header
                names = row
                mask = 1
            else:
                # judge the existance
                if row[4] in dic_Text.keys():
                    dic_Text[row[4]].append(row)
                else: 
                    dic_Text[row[4]] = list()
                    dic_Text[row[4]].append(row)

    
    with open(class_spe,'wb') as dfile:
        pickle.dump(dic_Text,dfile)
else:
    # divide the training set and the test set
    with open(class_spe,'rb') as dfile:
        dic_Text = pickle.load(dfile)
    
    Training_set_samples = dict()
    Test_set_samples = dict()
    for idx, label in enumerate(dic_Text.keys()):
        if idx == 100:
            break
        num_class = len(dic_Text[label])
        if num_class == 0:
            continue
        if num_class == 2 or num_class == 3:
            Training_set_samples[label] = dic_Text[label]
        else:
            idx_s = math.floor(num_class*0.7)  # It is perfect for the triplit loss design
            Training_set_samples[label] = dic_Text[label][:idx_s]
            Test_set_samples[label] = dic_Text[label][idx_s:]
    train_class_spe = 'train_divide_mini100_eq.pkl'
    test_class_spe = 'test_divide_mini100_eq.pkl'
    with open(train_class_spe,'wb') as dfile:
        pickle.dump(Training_set_samples,dfile)
    with open(test_class_spe,'wb') as dfile:
        pickle.dump(Test_set_samples,dfile)
