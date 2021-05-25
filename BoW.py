from sklearn.cluster import KMeans,MiniBatchKMeans
import numpy as np
from torch.utils.data.sampler import BatchSampler


class myBoW():
    def __init__(self,feas_train,labels_train,n_clusters):
        self.n_clusters = n_clusters
        self.feas_train = feas_train # feas_train中每个元素对应一个样本的所有向量
        self.labels_train = labels_train

        #获取所有样本向量的集合
        feas_All = list()
        idx_all = list()
        startIdx = 0
        endIdx = 0
        for n,feas in enumerate(feas_train):
            endIdx = startIdx + len(feas) #终止位
            idx_all.append((startIdx,endIdx))

            #展开
            for fea in feas:
                feas_All.append(fea)
            
            startIdx = endIdx #起始位
        
        #聚类
        # self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feas_All)
        self.kmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,batch_size=100,n_init=10, max_no_improvement=10, verbose=0)
        self.kmeans.fit(feas_All)



        #计算每个训练样本的词频向量（TF）
        self.WF_train = list()
        clusters_all = self.kmeans.labels_
        for n in range(len(feas_train)):
            startIdx = idx_all[n][0]
            endIdx = idx_all[n][1]
            clusters = clusters_all[startIdx:endIdx]
            self.WF_train.append(self.__WordFrequenncy__(clusters))

        #计算逆文档词频
        self.IDF = np.zeros(n_clusters)
        for k in range(n_clusters):
            for n in range(len(feas_train)):
                startIdx = idx_all[n][0]
                endIdx = idx_all[n][1]
                clusters = clusters_all[startIdx:endIdx]
                if(k in clusters):
                    self.IDF[k] = self.IDF[k] + 1
        self.IDF = np.log(len(feas_train)/self.IDF)

        #计算TF-IDF
        self.TF_IDF = list()
        for TF in self.WF_train:
            temp = TF * self.IDF
            self.TF_IDF.append(temp/np.linalg.norm(temp))



    def predict(self,feas):
        clusters = list()
        #计算词频向量
        for fea in feas:
            cluster = self.kmeans.predict([fea])
            clusters.append(cluster[0])

        TF = self.__WordFrequenncy__(clusters)

        #计算TF-IDF
        TF_IDF = TF*self.IDF/np.linalg.norm(TF*self.IDF)

        #计算与训练集合的相似度
        Max_simlarity_5 = np.zeros(5) #big to small 
        Best_label_5 = ['' for n in range(5)]
        for n,vec in enumerate(self.TF_IDF):
            simlarity = np.dot(vec,TF_IDF)
            for k in range(5):
                if(Max_simlarity_5[k]<simlarity):
                    if(k<4):
                        Max_simlarity_5[k+1:] = Max_simlarity_5[k:-1]
                        Best_label_5[k+1:] = Best_label_5[k:-1]
                    Max_simlarity_5[k] = simlarity
                    Best_label_5[k] = self.labels_train[n]
                    break
        
        return Best_label_5


    def __WordFrequenncy__(self,clusters):
        '''
        统计词频
        '''
        vector = np.zeros(self.n_clusters)
        for cluster_num in clusters:
           vector[cluster_num] =  vector[cluster_num] + 1
        return vector/len(clusters)


    