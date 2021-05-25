import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_SIFT
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from BoW import myBoW




#初始化
image_dir = "./shopee-product-matching/train_images"
text_path = "./shopee-product-matching/train.csv"           
epoch_num = 1   #总样本循环次数
batch_size = 1  #训练时的一组数据的大小

#读取数据集,并取出10%作为mini数据集
train_dataset = mydataset_SIFT(image_dir=image_dir,text_path = text_path, is_train=True)
test_dataset = mydataset_SIFT(image_dir=image_dir,text_path = text_path, is_train=False)

#计算训练集的SIFT特征向量
feas_train = []
labels_train = []
train_loader = DataLoader(dataset = train_dataset,batch_size = 1,shuffle = True)
# for image,text in train_loader: #遍历每一组数据
for batch_data in tqdm(train_loader):
    image,text=batch_data
      
    #处理图像数据，提取SIFT特征
    img = np.squeeze(image.numpy())
    img = np.transpose(img, (1,2,0))
    img = np.uint8((img + 1)*255/2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # changed into the gray form
    sift = cv2.xfeatures2d.SIFT_create() #创建一个SIFT对象
    # keyPoints = sift.detect(gray, None) 
    keyPoints, features = sift.detectAndCompute(gray, None)#检测特征点
    feas_train.append(features)
    labels_train.append(text[0]) # only given the label # TODO: text with a expand dimention



#训练词袋模型
model = myBoW(feas_train,labels_train,900)

#获取测试向量
#读取数据集,并取出10%作为mini数据集
train_dataset = mydataset_SIFT(image_dir=image_dir,text_path = text_path, is_train=True)
test_dataset = mydataset_SIFT(image_dir=image_dir,text_path = text_path, is_train=False)

#计算训练集的SIFT特征向量
feas_test = []
labels_test = []
results_image = {'pre':[],'GT':[]}
test_loader = DataLoader(dataset = test_dataset,batch_size = 1,shuffle = False)
# for image,text in train_loader: #遍历每一组数据
for batch_data in tqdm(test_loader):
    image,text=batch_data
    
    #处理图像数据，提取SIFT特征
    img = np.squeeze(image.numpy())
    img = np.transpose(img, (1,2,0))
    img = np.uint8((img + 1)*255/2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # changed into the gray form
    sift = cv2.xfeatures2d.SIFT_create() #创建一个SIFT对象
    # keyPoints = sift.detect(gray, None) 
    keyPoints, features = sift.detectAndCompute(gray, None)#检测特征点
    feas_test.append(features)
    labels_test.append(text[0]) # only given the label # TODO: text with a expand dimen

#用词袋模型进行测试
acc5 = 0
acc1 = 0
for n,feas in enumerate(feas_test):
    label_pre_5 = model.predict(feas)

    if(labels_test[n] == label_pre_5[0]):
        acc1 = acc1 + 1
    if(labels_test[n] in label_pre_5):
        acc5 = acc5 + 1

    label_pre_5 = np.flipud(label_pre_5) 
    results_image['pre'].append(label_pre_5)
    results_image['GT'].append(labels_test[n])

#save results
savePath = './results/sift_BoW_pure.pkl'
with open(savePath,'wb') as dfile: #Save dic to loacl
        pickle.dump(results_image,dfile)


acc_rate5 = acc5 / len(test_dataset)
acc_rate1 = acc1 / len(test_dataset)
print('----------------------------')
print('SIFT+BoW：')
# print(f"err = {err_rate:.4f}")
print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")
