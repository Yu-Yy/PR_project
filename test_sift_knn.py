import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_SIFT
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from BoW import myBoW
import visualize 


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


#计算测试集的SIFT特征向量，并匹配
test_loader = DataLoader(dataset = test_dataset,batch_size = 1,shuffle = False)
acc5 = 0
acc1 = 0
results_image = {'pre':[],'GT':[]}
# for image,text in train_loader: #遍历每一组数据
for batch_data in tqdm(test_loader):
    image,text=batch_data

    #处理图像数据，提取SIFT特征
    img = np.squeeze(image.numpy())
    img = np.transpose(img, (1,2,0))
    img = np.uint8((img + 1)*255/2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create() #创建一个SIFT对象
    # keyPoints = sift.detect(gray, None) 
    keyPoints, features = sift.detectAndCompute(gray, None)#检测特征点

    Max_Num_matches = 0
    top5_num_matches = np.zeros(5)
    pre_label = np.zeros(15)
    pre_label.dtype = '<U5'
    pre_label = pre_label[:5] 
    # do the R@5 test
    for i in range(len(feas_train)): #
        x = feas_train[i]
        y = labels_train[i]
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(x, features, k=2)
        good_Matches = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

        num_cat = np.concatenate([top5_num_matches,np.array([len(good_Matches)])])
        sorted_num = np.sort(num_cat)
        if sorted_num[0] != len(good_Matches):
            top5_num_matches = sorted_num[1:]
            pre_label = np.concatenate([pre_label,np.array([y])])
            idx_sort = np.argsort(num_cat)
            pre_label = pre_label[idx_sort]
            pre_label = pre_label[1:]
        # if(len(good_Matches)>Max_Num_matches):
        #     Max_Num_matches = len(good_Matches)
        #     pre_label = y
    results_image['pre'].append(pre_label)
    results_image['GT'].append(text[0])
    if text[0] in pre_label: #TODO: text need to be further index
        acc5 = acc5 + 1
    if text[0] == pre_label[-1]:
        acc1 = acc1 + 1
# err_rate = err/len(test_dataset)
# acc = 1-err_rate
acc_rate5 = acc5 / len(test_dataset)
acc_rate1 = acc1 / len(test_dataset)
print('----------------------------')
print('SIFT+knnMatch：')
# print(f"err = {err_rate:.4f}")
print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")

#save results
savePath = './results/test_SIFT_KNN.pkl'
with open(savePath,'wb') as dfile: #Save dic to loacl
        pickle.dump(results_image,dfile)

#visualize
cm = visualize.cal_confusion_matrix(results_image['pre'],results_image['GT'])
visualize.plot_confusion_matrix(cm, [], "SIFT KNN Confusion Matrix")
plt.savefig('./results/figures/SIFT KNN Confusion Matrix.jpg', format='jpg')