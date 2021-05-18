import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_NMF, mydataset_PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sklearn.decomposition as dc

# some bugs in the code resulting in low performance?


image_dir = "../shopee-product-matching/train_images"
text_path = "../shopee-product-matching/train.csv"           
epoch_num = 1   #总样本循环次数
batch_size = 1  #训练时的一组数据的大小

#读取数据集,并取出10%作为mini数据集 only test the easy mode
image_shape = (256,256)
train_dataset = mydataset_NMF(image_dir=image_dir,text_path = text_path, is_train=True)
test_dataset = mydataset_NMF(image_dir=image_dir,text_path = text_path, is_train=False)


feas_train = [] # create the dataset feature base in low dimension
labels_train = []
train_loader = DataLoader(dataset = train_dataset,batch_size = 1,shuffle = True)

nmf_estimator = dc.NMF(n_components=100, init='nndsvda', tol=5e-3, max_iter=1000)
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
    labels_train.extend([text[0] for _ in range(img_num)])
    # labels_train.append(text[0])

# do the NMF
labels_train = np.array(labels_train)
img_train = np.concatenate(img_train,axis=0)
# import pdb; pdb.set_trace()
trainned_base = nmf_estimator.fit_transform(img_train)
components_ = nmf_estimator.components_
# do the test
test_loader = DataLoader(dataset = test_dataset,batch_size = 1,shuffle = True)
acc5 = 0
acc1 = 0

for batch_data in tqdm(test_loader):
    image,text=batch_data

    #处理图像数据，提取SIFT特征
    # img_test = np.squeeze(image.numpy())
    img_test = image.numpy()
    img_test = img_test[0,...]
    # img_c = img_test.reshape(1,-1)
    img_feature = nmf_estimator.transform(img_test)

    # import pdb; pdb.set_trace()
    # recovered = img_feature @ components_
    # vmax = max(recovered.max(), -recovered.min())
    # plt.imshow(recovered.reshape(image_shape), cmap=plt.cm.gray,
    #                interpolation='nearest',
    #                vmin=-vmax, vmax=vmax)
    # plt.show()

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
