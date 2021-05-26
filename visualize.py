from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import tensorflow as tf
import requests
import pickle




def cal_confusion_matrix(pre_List,GT_list):
    GT_list = np.array(GT_list)
    label_list = np.sort(np.unique(GT_list))

    confusion_matrix = np.zeros((len(label_list),len(label_list)))

    for i,label in enumerate(label_list):
        pre_all = np.array([item[-1] for item in  pre_List])
        index_list  = np.where(pre_all == label)[0]
        for index in index_list:
            j = np.where(label_list == GT_list[index])[0]
            confusion_matrix[i][j] += 1

    return confusion_matrix



def plot_confusion_matrix(cm, labels_name, title):
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
    # cm = [[100,1,0 ,1 ,  6 ,  0 ,  0],
    # [2 ,111  , 3 ,  0 ,  2 ,  1 , 24],
    # [  0  , 2 , 68 ,  5  , 4   ,3 ,  2],
    # [  2  , 0  , 1 ,120 ,  7 , 26 ,  0],
    # [  2  , 5  , 3 ,  2 ,120  ,11 , 14],
    # [  2  , 0  , 2  ,12  , 8 ,115  , 1],
    # [  2  ,25 ,  0  , 1 , 14 ,  4 ,302]]
    # cm = np.array(cm)
    # result_file = './results/test_SIFT_KNN.pkl'
    result_file = './results/test_text_pure.pkl'
    
    with open(result_file,'rb') as f:
        results = pickle.load(f)

    cm = cal_confusion_matrix(results['pre'],results['GT'])
    plot_confusion_matrix(cm, [], "text pure Confusion Matrix")
    plt.savefig('./results/figures/text pure Confusion Matrix.jpg', format='jpg')