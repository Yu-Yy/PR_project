from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import tensorflow as tf
import requests
import pickle
from  resultAyalyse import cal_confusion_matrix


def drawDistribution():
    '''
    画出数据的分布（直方图）
    '''
    full_dataset = './datasets/class_specific.pkl'
    train_class_spe = './datasets/train_divide_mini500_eq.pkl'
    val_class_spe = './datasets/val_divide_mini500_eq.pkl'
    test_class_spe = './datasets/test_divide_mini500_eq.pkl'

    #load full data
    with open(full_dataset,'rb') as dfile: #read dic
        dic_full = pickle.load(dfile)
    label_list = [item for item in dic_full.keys()]
    His_full = np.zeros(len(label_list))
    for n,label in enumerate(label_list):
        His_full[n] += len(dic_full[label])


    #load train data
    with open(train_class_spe,'rb') as dfile: #read dic
        dic_train = pickle.load(dfile)
    train_list = [item for item in dic_train.keys()]
    His_train = np.zeros(500)
    for label in train_list:
        n = label_list.index(label)
        His_train[n] += len(dic_train[label])


    #load val data
    with open(val_class_spe,'rb') as dfile: #read dic
        dic_val = pickle.load(dfile)
    val_list = [item for item in dic_val.keys()]
    His_val = np.zeros(500)
    for label in val_list:
        n = label_list.index(label)
        His_val[n] += len(dic_val[label])

    #load test data
    with open(val_class_spe,'rb') as dfile: #read dic
        dic_test = pickle.load(dfile)
    test_list = [item for item in dic_test.keys()]
    His_test = np.zeros(500)
    for label in test_list:
        n = label_list.index(label)
        His_test[n] += len(dic_test[label])

    # train1 = dic_train['2518689634']
    # test1 = dic_test['2518689634']
    # val1 = dic_val['2518689634']
    #draw distritutionfigsize=(20,6)
    # fig, ax = plt.subplots(ncols=2,nrows=1)
    plt.figure()
    X = range(len(label_list))
    plt.bar(X, His_full, width=0.8,color='b')
    plt.title('Distribution of full dataset')
    plt.xlabel('classes')
    plt.ylabel('number of samples')
    plt.savefig("./results/figures/Histogram of full dataset.jpg", format='jpg')
 

    plt.figure()
    X = range(500)
    plt.bar(X, His_train, width=0.8,color='b')
    plt.bar(X, His_val, width=0.8,color='g',bottom=His_train)
    plt.bar(X, His_test, width=0.8,color='y',bottom=(His_val+His_train))
    plt.legend(['Train','Validation','Test'])
    plt.title('Distribution of mini dataset')
    plt.xlabel('classes')
    plt.ylabel('number of samples')

    #save fig
    plt.savefig("./results/figures/Histogram of mini dataset.jpg", format='jpg')







def plot_confusion_matrix(cm, labels_name, title):
    plt.figure()
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+1e-6)    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')


if __name__=='__main__':

    drawDistribution()

    import os
    filePath = './results'
    fileList = os.listdir(filePath)

    for t,filename in enumerate(fileList):
        if not '.pkl' in filename:
            continue

        result_file = '/'.join([filePath,filename])
        with open(result_file,'rb') as f:
            results = pickle.load(f)
            cm = cal_confusion_matrix(results['pre'],results['GT'])

            figname = filename.replace('.pkl', '')
            figname = figname.replace('test', '')
            figname = figname.replace('_', ' ')
            figname = figname.replace('result', '')+' Confusion Matrix'
            plot_confusion_matrix(cm, [], figname)
            plt.savefig('./results/figures/'+figname+'.jpg', format='jpg')
    

        

   
    
    
    