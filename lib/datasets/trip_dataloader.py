import copy
import logging

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import pickle
import os.path as osp

BOG_file = 'datasets/Total_BoW.pkl'
with open(BOG_file,'rb') as f:
    BOG = pickle.load(f)
BOG_list = list(BOG)
Text_dim = len(BOG_list)

TEST = 'datasets/test_divide_eq.pkl'  # using the whole dataset to train the twins network
TRAIN = 'datasets/train_divide_eq.pkl'

TEST_V = 'datasets/test_divide_mini500_eq.pkl'
TRAIN_V = 'datasets/train_divide_mini500_eq.pkl'

class trip_retrieval(Dataset):  # Try to using simple metric learning loss via classification comparasion
    def __init__(self, image_folder, is_train = True, resize_height = 640, resize_width = 640):
        super(trip_retrieval,self).__init__()
        # self.transforms=True
        self.size = (resize_height, resize_width)
        self.transform= transforms.Compose([
        #  transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #
        ])
        self.image_folder = image_folder
        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        
        with open(self.dataset,'rb') as f:
            self.raw_data = pickle.load(f)
        # get the total length via class number
        self.length = len(self.raw_data)
        self.label_list = list(self.raw_data)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        selected_data = self.raw_data[self.label_list[i]]
        selected_length = len(selected_data)
        # choose one as q and else as positive g
        random_idx = torch.randperm(selected_length)[:2]
        q_frame = cv2.imread(osp.join(self.image_folder,self.raw_data[self.label_list[i]][random_idx[0]][1]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        q_frame = cv2.resize(q_frame,self.size, interpolation = cv2.INTER_AREA)
        g_frame = cv2.imread(osp.join(self.image_folder,self.raw_data[self.label_list[i]][random_idx[1]][1]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        g_frame = cv2.resize(g_frame,self.size, interpolation=cv2.INTER_AREA)

        q_img = self.transform(q_frame)
        g_img = self.transform(g_frame)

        return q_img, g_img


class val_retrieval(Dataset):  # Try to using simple metric learning loss via classification comparasion
    def __init__(self, image_folder, is_train = True, resize_height = 640, resize_width = 640):
        super(val_retrieval,self).__init__()
        # self.transforms=True
        self.size = (resize_height, resize_width)
        self.transform= transforms.Compose([
        #  transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #
        ])
        self.image_folder = image_folder
        if is_train:
            self.dataset = TRAIN_V
        else:
            self.dataset = TEST_V
        
        with open(self.dataset,'rb') as f:
            self.raw_data = pickle.load(f)
        # get the total length via class number
        self.imgs = list()
        self.texts = list()
        for l,v in self.raw_data.items():
            self.imgs.extend(v)
            self.texts.extend([l for _ in range(len(v))])
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img_path = os.path.join(self.image_folder,self.imgs[i][1])

        q_frame = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        q_frame = cv2.resize(q_frame,self.size, interpolation = cv2.INTER_AREA)
        # g_frame = cv2.imread(osp.join(self.image_folder,self.raw_data[self.label_list[i]][random_idx[1]][1]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # g_frame = cv2.resize(g_frame,self.size, interpolation=cv2.INTER_AREA)

        q_img = self.transform(q_frame)
        
        label = self.texts[i]
        
        return q_img ,label


class triplet_image_text_data(Dataset):
    def __init__(self,image_folder, is_train = True, resize_height = 640, resize_width = 640):
        super(triplet_image_text_data,self).__init__()
        self.size = (resize_height, resize_width)
        self.transform= transforms.Compose([
        #  transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #
        ])
        self.image_folder = image_folder
        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        with open(self.dataset,'rb') as f:
            self.raw_data = pickle.load(f)
        # get the total length via class number
        self.length = len(self.raw_data)
        self.label_list = list(self.raw_data)
    def __len__(self):
        return self.length

    def __getitem__(self,i):
        selected_data = self.raw_data[self.label_list[i]]
        selected_length = len(selected_data)
        random_idx = torch.randperm(selected_length)[:2]

        # get the image info
        q_frame = cv2.imread(osp.join(self.image_folder,self.raw_data[self.label_list[i]][random_idx[0]][1]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        q_frame = cv2.resize(q_frame,self.size, interpolation = cv2.INTER_AREA)
        g_frame = cv2.imread(osp.join(self.image_folder,self.raw_data[self.label_list[i]][random_idx[1]][1]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        g_frame = cv2.resize(g_frame,self.size, interpolation=cv2.INTER_AREA)

        q_img = self.transform(q_frame)
        g_img = self.transform(g_frame)

        # get the text info
        q_feature = torch.ones(Text_dim,1) + 1e-5
        q_words = self.raw_data[self.label_list[i]][random_idx[0]][3].split()
        for word in q_words:
            if word.isalpha():
                word = word.upper()
                if word in BOG_list:
                    idx = BOG_list.index(word)
                    q_feature[idx,:] = q_feature[idx,:] + 1
        q_feature = torch.log(q_feature)

        g_feature = torch.ones(Text_dim,1) + 1e-5
        g_words = self.raw_data[self.label_list[i]][random_idx[1]][3].split()
        for word in g_words:
            if word.isalpha():
                word = word.upper()
                if word in BOG_list:
                    idx = BOG_list.index(word)
                    g_feature[idx,:] = g_feature[idx,:] + 1
        g_feature = torch.log(g_feature)

        return q_img, g_img, q_feature, g_feature



class text_retrieval(Dataset):
    def __init__(self,image_folder, is_train = True):
        super().__init__()
        # self.size = (resize_height, resize_width)
        self.transform= transforms.Compose([
        #  transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #
        ])
        self.image_folder = image_folder
        if is_train:
            self.dataset = TRAIN_V
        else:
            self.dataset = TEST_V
        with open(self.dataset,'rb') as f:
            self.raw_data = pickle.load(f)
        
        self.text = list()
        self.label = list()
        for l,v in self.raw_data.items():
            # process the text information
            for v_i in v:
                words = v_i[3].split()
                new_wordlist = []
                for word in words:
                    if word.isalpha():
                        word = word.upper()
                        new_wordlist.append(word)
                # append in txt list
                self.text.append(new_wordlist)

            self.label.extend([l for _ in range(len(v))])
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self,i):

        text_feature = torch.ones(Text_dim,1) + 1e-5 
        wordlist = self.text[i]
        # create the word feature
        for word in wordlist:
            if word in BOG_list:
                idx = BOG_list.index(word)
                text_feature[idx,:] += 1.0
        text_feature = torch.log(text_feature)  # for non-neg
        label = self.label[i]

        return text_feature, label

class image_text_eval(Dataset):
    def __init__(self, image_folder, is_train = True,resize_height = 640, resize_width = 640):
        super().__init__()
        self.size = (resize_height, resize_width)
        self.transform= transforms.Compose([
        #  transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #
        ])
        self.image_folder = image_folder
        if is_train:
            self.dataset = TRAIN_V
        else:
            self.dataset = TEST_V
        
        with open(self.dataset,'rb') as f:
            self.raw_data = pickle.load(f)
        # get the total length via class number
        self.imgs = list()
        self.label = list()
        self.text = list()
        for l,v in self.raw_data.items():
            self.imgs.extend(v)
            self.label.extend([l for _ in range(len(v))])
            for v_i in v:
                words = v_i[3].split()
                new_wordlist = []
                for word in words:
                    if word.isalpha():
                        word = word.upper()
                        new_wordlist.append(word)
                # append in txt list
                self.text.append(new_wordlist)
        
    def __len__(self):
        return len(self.label)
    def __getitem__(self, i):
        img_path = os.path.join(self.image_folder,self.imgs[i][1])

        q_frame = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        q_frame = cv2.resize(q_frame,self.size, interpolation = cv2.INTER_AREA)
        # g_frame = cv2.imread(osp.join(self.image_folder,self.raw_data[self.label_list[i]][random_idx[1]][1]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # g_frame = cv2.resize(g_frame,self.size, interpolation=cv2.INTER_AREA)

        q_img = self.transform(q_frame)
        # text feature
        text_feature = torch.ones(Text_dim,1) + 1e-5 
        wordlist = self.text[i]
        # create the word feature
        for word in wordlist:
            if word in BOG_list:
                idx = BOG_list.index(word)
                text_feature[idx,:] += 1.0
        text_feature = torch.log(text_feature)  # for non-neg
        label = self.label[i]

        return q_img, text_feature, label