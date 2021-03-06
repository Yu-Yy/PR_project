import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sklearn.decomposition as dc
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--aug', help='if processing data augmentation or not', required=True, default=False ,type=bool)
args = parser.parse_args()

image_dir = "shopee-product-matching/train_images" 
text_path = "shopee-product-matching/train.csv"  

epoch_num = 1   #总样本循环次数
batch_size = 1  #训练时的一组数据的大小

#读取数据集,并取出10%作为mini数据集 only test the easy mode
train_dataset = mydataset_PCA(image_dir=image_dir,text_path = text_path, is_train=True, is_augmentation=args.aug)
test_dataset = mydataset_PCA(image_dir=image_dir,text_path = text_path, is_train=False)

feas_train = [] # create the dataset feature base in low dimension
labels_train = []
train_loader = DataLoader(dataset = train_dataset,batch_size = 1,shuffle = True)

pca_estimator = dc.PCA(n_components=100)
# for image,text in train_loader: #遍历每一组数据
img_train = []
for batch_data in tqdm(train_loader):
    image,text=batch_data

    # img = np.squeeze(image.numpy())
    img = image.numpy()
    img = img[0,...]
    # only using the
    # img_train.append(img.reshape(1,-1))
    img_train.append(img)
    img_num = img.shape[0]
    # labels_train.append(text[0])
    labels_train.extend([text[0] for _ in range(img_num)])

# do the PCA
labels_train = np.array(labels_train)
img_train = np.concatenate(img_train,axis=0)
img_mean = np.mean(img_train, axis=0, keepdims=True)
img_train = img_train - img_mean
trainned_base = pca_estimator.fit_transform(img_train)
components_ = pca_estimator.components_
# do the test
test_loader = DataLoader(dataset = test_dataset,batch_size = 1,shuffle = True)
acc5 = 0
acc1 = 0

for batch_data in tqdm(test_loader):
    image,text=batch_data
    #处理图像数据，提取SIFT特征
    # img_test = np.squeeze(image.numpy())
    img_test = image.numpy()
    # img_c = img.reshape(1,-1)
    img_c = img_test[0,...]
    img_c = img_c - img_mean
    img_feature = pca_estimator.transform(img_c)
    distance_s = np.sum((img_feature - trainned_base) ** 2, axis=-1)
    idx_sort = np.argsort(distance_s)
    idx_top5 = idx_sort[:5]
    pred_label = labels_train[idx_top5]

    if text[0] in pred_label: #TODO: text need to be further index
        acc5 = acc5 + 1
    if text[0] == pred_label[0]:
        acc1 = acc1 + 1

# err_rate = err/len(test_dataset)
# acc = 1-err_rate
acc_rate5 = acc5 / len(test_dataset)
acc_rate1 = acc1 / len(test_dataset)
print('----------------------------')
# print(f"err = {err_rate:.4f}")
print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")
