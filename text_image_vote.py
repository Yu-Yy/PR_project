import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_SIFT
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from BoW import myBoW

#load results:
result_sift_knn_file = './results/test_SIFT_KNN.pkl'
result_text_pure_file = './results/test_text_pure.pkl'
with open(result_sift_knn_file,'rb') as f:
    results_image = pickle.load(f)
with open(result_text_pure_file,'rb') as f:
    results_text = pickle.load(f)
N_samples = len(results_image['pre'])

#Check test_SIFT_KNN
acc1 = 0
acc5 = 0
for n,pre_label in enumerate(results_image['pre']):
    GT = results_image['GT'][n]
    if GT in pre_label: #TODO: text need to be further index
        acc5 = acc5 + 1
    if GT == pre_label[-1]:
        acc1 = acc1 + 1
acc_rate5 = acc5 / N_samples
acc_rate1 = acc1 / N_samples
print('----------------------------')
print('sift+knn：')
print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")

#Check test_text_pure
acc1 = 0
acc5 = 0
for n,pre_label in enumerate(results_text['pre']):
    GT = results_text['GT'][n]
    if GT in pre_label: #TODO: text need to be further index
        acc5 = acc5 + 1
    if GT == pre_label[-1]:
        acc1 = acc1 + 1
acc_rate5 = acc5 / N_samples
acc_rate1 = acc1 / N_samples
print('----------------------------')
print('text_pure：')
print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")

#Fusion
acc_Borda = 0
acc_vote = 0
acc5_fusion = 0
for n in range(len(results_text['pre'])):
    #Borda Count
    preList = np.unique(np.hstack((results_image['pre'][n],results_text['pre'][n])))
    socreList = np.zeros(len(preList))
    for i in range(5):
        socreList[np.where(preList == results_image['pre'][n][i])] += i
        socreList[np.where(preList == results_text['pre'][n][i])] += i
    bestLabel = preList[np.argmax(socreList)]
    if bestLabel == results_text['GT'][n]:
        acc_Borda += 1


    #Vote
    pre = np.hstack((results_image['pre'][n],results_text['pre'][n])).tolist()
    maxlabel = max(pre,key=pre.count)
    if(maxlabel==results_text['GT'][n]):
        acc_vote = acc_vote + 1
    if(results_text['GT'][n] in pre):
        acc5_fusion = acc5_fusion + 1
acc_rate_vote = acc_vote / len(results_text['pre'])
acc_rate5_fusion = acc5_fusion / len(results_text['pre'])
acc_rate_Borda = acc_Borda / len(results_text['pre'])
print('----------------------------')
print('Text_Image_fusion:')
print(f"acc_vote = {acc_rate_vote:.4f}")
print(f"acc_rate_Borda = {acc_rate_Borda:.4f}")
print(f"acc5_fusion = {acc_rate5_fusion:.4f}")
        

        
        


    





