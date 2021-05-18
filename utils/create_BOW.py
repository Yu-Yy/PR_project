import numpy as np
import pickle
import csv
import re

text_path = "../shopee-product-matching/train.csv"  
Total_bow = 'Total_BoW.pkl'

creating_BOW = True
if creating_BOW: # create the Bag of words for specific dataset including 
    dic_Text = {}
    with open(text_path, 'r') as f:
        reader = csv.reader(f)
        mask = 0
        for row in reader:
            if mask == 0: # gap the header
                names = row
                mask = 1
            else:
                # create the words list and ignore the capitilized word
                word_list = row[3].split()
                for word in word_list:
                    # if bool(re.search(r'\d', word)):
                    #     pass
                    if word.isalpha():
                        word = word.upper() # standard
                        if word in dic_Text.keys():
                            dic_Text[word] += 1
                        else:
                            dic_Text[word] = 1
        # delete the redundent words
        dic_copy = dic_Text.copy()
        for k,v in dic_Text.items():
            if dic_Text[k] == 1:
                # 删除该词
                del dic_copy[k]

    
    with open(Total_bow,'wb') as dfile:
        pickle.dump(dic_copy,dfile)