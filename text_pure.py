# import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import Text_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sklearn.decomposition as dc

is_pca = False
is_nmf = False

image_dir = "../shopee-product-matching/train_images"
text_path = "../shopee-product-matching/train.csv"           
epoch_num = 1   #总样本循环次数
batch_size = 1  #训练时的一组数据的大小

#读取数据集,并取出10%作为mini数据集 only test the easy mode
train_dataset = Text_dataset(image_dir=image_dir, is_train=True)
test_dataset = Text_dataset(image_dir=image_dir, is_train=False)

if is_pca:
    dc_er = dc.PCA(n_components=100)
if is_nmf:
    dc_er = dc.NMF(n_components=100, init='nndsvda', tol=5e-3, max_iter=1000)

feas_train = [] # create the dataset feature base in low dimension
labels_train = []
train_loader = DataLoader(dataset = train_dataset,batch_size = 1,shuffle = True)

# pca_estimator = dc.PCA(n_components=100)
# for image,text in train_loader: #遍历每一组数据
for batch_data in tqdm(train_loader):
    text,label=batch_data
    text = text.numpy()
    text = text[0,...]
    label = label[0]
    feas_train.append(text)
    labels_train.append(label)

feas_train = np.concatenate(feas_train,axis=0)
# feas_mean = np.mean(feas_train, axis=0, keepdims=True)
# feas_train = feas_train - feas_mean

labels_train = np.array(labels_train)
trainned_base  = feas_train
# trainned_base = dc_er.fit_transform(feas_train)

# do the test
test_loader = DataLoader(dataset = test_dataset,batch_size = 1,shuffle = False)
acc5 = 0
acc1 = 0
results_image = {'pre':[],'GT':[]}
for batch_data in tqdm(test_loader):
    text,label = batch_data
    #处理图像数据，提取SIFT特征
    # img_test = np.squeeze(image.numpy())
    text_test = text.numpy()
    # img_c = img.reshape(1,-1)
    text_test = text_test[0,...]
    # text_test = text_test - feas_mean
    # text_feature = dc_er.transform(text_test)
    text_feature = text_test
    distance_s = np.sum((text_feature - trainned_base) ** 2, axis=-1)
    idx_sort = np.argsort(distance_s)
    idx_top5 = idx_sort[:5]
    pred_label = labels_train[idx_top5]
    results_image['pre'].append(pred_label)
    results_image['GT'].append(label[0])
    if label[0] in pred_label: #TODO: text need to be further index
        acc5 = acc5 + 1
    if label[0] == pred_label[0]:
        acc1 = acc1 + 1

file_name = 'pred_result/pure_text_result.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(results_image , f)

# err_rate = err/len(test_dataset)
# acc = 1-err_rate
acc_rate5 = acc5 / len(test_dataset)
acc_rate1 = acc1 / len(test_dataset)
print('----------------------------')

print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")