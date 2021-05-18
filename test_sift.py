import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_SIFT
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

#初始化
image_dir = "../shopee-product-matching/train_images"
text_path = "../shopee-product-matching/train.csv"           
epoch_num = 1   #总样本循环次数
batch_size = 1  #训练时的一组数据的大小

#读取数据集,并取出10%作为mini数据集
train_dataset = mydataset_SIFT(image_dir=image_dir,text_path = text_path, is_train=True)
test_dataset = mydataset_SIFT(image_dir=image_dir,text_path = text_path, is_train=False)
# mini_size = 1000  #int(0.25* len(train_dataset))
# left_size = len(train_dataset) - mini_size
# mini_train_data, left_train_data = torch.utils.data.random_split(train_dataset, [mini_size, left_size])

# mini_size = 500   #int(0.25* len(train_dataset))
# left_size = len(test_dataset) - mini_size
# mini_test_data, left_test_data = torch.utils.data.random_split(test_dataset, [mini_size, left_size])
# #随机划分训练集、验证集和测试集
# train_size = int(0.6 * mini_size)
# val_size = int(0.2 * mini_size)
# test_size = mini_size - train_size - val_size
# train_dataset, val_dataset,test_dataset = torch.utils.data.random_split(mini_train_data, [train_size, val_size,test_size])

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


    #显示SIFT特征点
    # showimg = img.copy()
    # cv2.drawKeypoints(gray, keyPoints, showimg)
    # # cv2.drawKeypoints(gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(showimg)
    # plt.show()
    # plt.savefig('result.jpg')

# # store the features
# train_dic_file = 'train_feature.pkl'
# train_label_file = 'train_label.pkl'
# with open(train_dic_file,'wb') as dfile:
#     pickle.dump(feas_train,dfile)
# with open(train_label_file,'wb') as dfile:
#     pickle.dump(labels_train,dfile)


#计算测试集的SIFT特征向量，并匹配
test_loader = DataLoader(dataset = test_dataset,batch_size = 1,shuffle = True)
acc5 = 0
acc1 = 0
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
    if text[0] in pre_label: #TODO: text need to be further index
        acc5 = acc5 + 1
    if text[0] == pre_label[-1]:
        acc1 = acc1 + 1
# err_rate = err/len(test_dataset)
# acc = 1-err_rate
acc_rate5 = acc5 / len(test_dataset)
acc_rate1 = acc1 / len(test_dataset)
print('----------------------------')
# print(f"err = {err_rate:.4f}")
print(f"acc1 = {acc_rate1:.4f}")
print(f"acc5 = {acc_rate5:.4f}")

        
        


    




# img = cv2.imread('./shopee-product-matching/test_images/0006c8e5462ae52167402bac1c2e916e.jpg')
# img1 = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# sift = cv2.xfeatures2d.SIFT_create() #创建一个SIFT对象
# kp = sift.detect(gray, None) #检测特征点
 
# #绘制特征点
# cv2.drawKeypoints(gray, kp, img) 
# cv2.drawKeypoints(gray, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# plt.subplot(121), plt.imshow(img),
# plt.title('Dstination'), plt.axis('off')
# plt.subplot(122), plt.imshow(img1),
# plt.title('Dstination'), plt.axis('off')
# plt.show()

