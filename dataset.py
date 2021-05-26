import csv
import random
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os,sys
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import cv2
from utils.data_augmentation import DataAugmentation


BOW_file = 'datasets/Total_BoW.pkl'
with open(BOW_file,'rb') as f:
    BOW = pickle.load(f)
BOW_list = list(BOW)
Text_dim = len(BOW_list)

TEST = 'datasets/test_divide_mini500_eq.pkl' 
TRAIN = 'datasets/train_divide_mini500_eq.pkl'

class mydataset_SIFT(Dataset):
    def __init__(self, image_dir,text_path,resize_height=640, resize_width=640,is_train = True):
        '''
        Input:  image_dir -图片路径(image_dir+imge_name.jpg构成图片的完整路径)
                text_path - 文本数据的路径
                resize_height -图像高，
                resize_width  -图像宽    
        '''

        # #相关预处理的初始化
        self.transforms=True
        self.transform= transforms.Compose([
         transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #
        ])

        self.image_dir = image_dir
        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        with open(self.dataset,'rb') as dfile:
            raw_data = pickle.load(dfile)
        self.imgs = list()
        self.texts = list()
        for l,v in raw_data.items():
            self.imgs.extend(v)
            self.texts.extend([l for _ in range(len(v))])
       
 
    def __getitem__(self, i):
        #获取图像
        img_path = os.path.join(self.image_dir,self.imgs[i][1])
        pil_img = Image.open(img_path)
        if self.transforms:
            img_data =self.transform(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            img_data = torch.from_numpy(pil_img)
        
        #获取文本
        text_data = self.texts[i]

        #返回数据
        return [img_data,text_data]
 
    def __len__(self):
        return len(self.imgs)
 
class mydataset_NMF(Dataset):
    def __init__(self, image_dir,text_path,resize_height=256, resize_width=256,is_train = True, is_augmentation = True):
        '''
        Input:  image_dir -图片路径(image_dir+imge_name.jpg构成图片的完整路径)
                text_path - 文本数据的路径
                resize_height -图像高，
                resize_width  -图像宽    
        '''

        # #相关预处理的初始化
        self.transforms=False
        self.transform= transforms.Compose([
         transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Do the data augmentation
        if is_train:
            if is_augmentation:
                self.augmentation = True
            else:
                self.augmentation = False
        else:
            self.augmentation = False
        self.augmentation_tool = DataAugmentation 
        self.augmentation_method = np.array(["randomRotation",
               "randomCrop",
               "randomColor",
               "randomGaussian"
        ])
        self.new_size = (resize_height,resize_width)
        self.image_dir = image_dir
        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        with open(self.dataset,'rb') as dfile:
            raw_data = pickle.load(dfile)
        self.imgs = list()
        self.texts = list()
        for l,v in raw_data.items():
            self.imgs.extend(v)
            self.texts.extend([l for _ in range(len(v))])
       
 
    def __getitem__(self, i):
        #获取图像
        img_path = os.path.join(self.image_dir,self.imgs[i][1])
        pil_img = Image.open(img_path)
        # pil_img = pil_img.convert('L') 
        # pil_img = cv2.imread(img_path,flags=cv2.IMREAD_GRAYSCALE)
        # do the augmentation
        pil_img = pil_img.resize(self.new_size)
        if self.augmentation:
            augmentation_img_collect =[]
            rand_method = np.random.randint(3,size=(9))
            for idx in rand_method:
                aug_img = eval('DataAugmentation.'+ self.augmentation_method[idx])(pil_img)
                aug_img = aug_img.resize(self.new_size) # keep the same size
                aug_img = aug_img.convert('L')
                aug_img = np.asarray(aug_img,dtype=np.float32) / 255
                augmentation_img_collect.append(aug_img.reshape(1,-1))
            augmentation_img_collect = np.concatenate(augmentation_img_collect,axis=0)

        pil_img = pil_img.convert('L') 
        pil_img = np.asarray(pil_img,dtype=np.float32)/255
        pil_img = pil_img.reshape(1,-1)
        if self.augmentation:
            pil_img = np.concatenate([pil_img, augmentation_img_collect],axis=0) 
        #获取文本
        img_data = torch.from_numpy(pil_img)
        text_data = self.texts[i]
        #返回数据
        return [img_data,text_data]
 
    def __len__(self):
        return len(self.imgs)   
 
class mydataset_PCA(Dataset):
    def __init__(self, image_dir,text_path,resize_height=256, resize_width=256,is_train = True, is_augmentation=True):
        '''
        Input:  image_dir -图片路径(image_dir+imge_name.jpg构成图片的完整路径)
                text_path - 文本数据的路径
                resize_height -图像高，
                resize_width  -图像宽    
        '''

        # #相关预处理的初始化
        self.transforms=False
        self.transform= transforms.Compose([
         transforms.Resize((resize_height,resize_width)),
         transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if is_train:
            if is_augmentation:
                self.augmentation = True
            else:
                self.augmentation = False
        else:
            self.augmentation = False

        self.augmentation_tool = DataAugmentation 
        self.augmentation_method = np.array(["randomRotation",
               "randomCrop",
               "randomColor",
               "randomGaussian"
        ])
        self.new_size = (resize_height,resize_width)
        self.image_dir = image_dir
        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        with open(self.dataset,'rb') as dfile:
            raw_data = pickle.load(dfile)
        self.imgs = list()
        self.texts = list()
        for l,v in raw_data.items():
            self.imgs.extend(v)
            self.texts.extend([l for _ in range(len(v))])
       
 
    def __getitem__(self, i):
        #获取图像
        img_path = os.path.join(self.image_dir,self.imgs[i][1])
        pil_img = Image.open(img_path)

         # do the augmentation
        pil_img = pil_img.resize(self.new_size)
        if self.augmentation:
            augmentation_img_collect =[]
            rand_method = np.random.randint(3,size=(9))
            for idx in rand_method:
                aug_img = eval('DataAugmentation.'+ self.augmentation_method[idx])(pil_img)
                aug_img = aug_img.resize(self.new_size) # keep the same size
                aug_img = aug_img.convert('L')
                aug_img = np.asarray(aug_img,dtype=np.float32) / 255
                augmentation_img_collect.append(aug_img.reshape(1,-1))
            augmentation_img_collect = np.concatenate(augmentation_img_collect,axis=0)

        pil_img = pil_img.convert('L') 
        pil_img = np.asarray(pil_img,dtype=np.float32)/255
        pil_img = pil_img.reshape(1,-1)
        if self.augmentation:
            pil_img = np.concatenate([pil_img, augmentation_img_collect],axis=0) 
        #获取文本
        img_data = torch.from_numpy(pil_img)
        text_data = self.texts[i]
        #返回数据
        return [img_data,text_data]

        # pil_img = pil_img.convert('L') 
        # # pil_img = cv2.imread(img_path,flags=cv2.IMREAD_GRAYSCALE)

        # if self.transforms:
        #     img_data =self.transform(pil_img)
        # else:
        #     pil_img = pil_img.resize(self.new_size)
        #     pil_img = np.asarray(pil_img,dtype=np.float32)
        #     pil_img = pil_img/255
        #     img_data = torch.from_numpy(pil_img)        
        # #获取文本
        # text_data = self.texts[i]
        # #返回数据
        # return [img_data,text_data]
 
    def __len__(self):
        return len(self.imgs)

# create the text indexing method
class Text_dataset(Dataset):
    def __init__(self,image_dir,is_train = True):
        super(Text_dataset,self).__init__()
        self.image_dir = image_dir
        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        with open(self.dataset,'rb') as dfile:
            raw_data = pickle.load(dfile)
        self.text = list()
        self.label = list()
        for l,v in raw_data.items():
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
        # transfer the word into the dimentsion
        
        text_feature = np.ones((1,Text_dim)) + 1e-5 
        wordlist = self.text[i]
        # create the word feature
        for word in wordlist:
            if word in BOW_list:
                idx = BOW_list.index(word)
                text_feature[:,idx] += 1.0
        text_feature = np.log(text_feature)  # for non-neg
        text_data = torch.from_numpy(text_feature)
        label = self.label[i]

        return text_data,label







if __name__=='__main__':
    image_dir = "../shopee-product-matching/train_images"
    text_path = "../shopee-product-matching/train.csv"           

 
    epoch_num=1   #总样本循环次数 反对
    batch_size=10  #训练时的一组数据的大小
    train_data_nums=13233
    max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数 

    train_data = mydataset_SIFT(image_dir=image_dir,text_path = text_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
 
    #使用epoch方法迭代，LfwDataset的参数repeat=1
    for epoch in range(epoch_num):
        for batch_data in train_loader:
            image,text=batch_data
            image=image.numpy()  
            plt.imshow(image)
            plt.show()
            print("batch_image.shape:{}".format(batch_data.shape))



