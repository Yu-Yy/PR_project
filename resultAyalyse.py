import numpy as np
import pickle
import os


TRAIN = 'datasets/train_divide_mini500_eq.pkl'

def cal_confusion_matrix(pre_List,GT_list):
    # GT_list = np.array(GT_list)
    # label_list = np.sort(np.unique(GT_list))

    with open(TRAIN,'rb') as f:
        dic_train = pickle.load(f)
    label_list = np.sort(np.unique(np.array([item for item in dic_train.keys()])))

    confusion_matrix = np.zeros((len(label_list),len(label_list)))
    pre_all = np.array([item[0] for item in  pre_List])

    for n,pre in enumerate(pre_all):
        
        if not pre  in label_list:
            continue
        
        i = np.where(label_list == pre)[0][0]
        GT = GT_list[n]
        j = np.where(label_list == GT)[0][0]
        confusion_matrix[i][j] += 1

    
    return confusion_matrix


def evaluate(cm):
    '''
    由混淆矩阵计算评价指标：pre,recall,F1
    '''
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    prelist = list()
    recalllist = list()
    f1list = list()
    W_list = list()
    for k in range(len(cm)):
        TP = cm[k,k]
        FP = np.sum(cm[k,:]) - TP
        FN = np.sum(cm[:,k]) - TP

        pre = TP/(TP+FP+1e-6)
        recall = TP/(TP+FN+1e-6)
        f1 = 2*pre*recall/(pre + recall+1e-6)

        if (TP ==0 and FP ==0 and FN == 0):
            continue

        prelist.append(pre)
        recalllist.append(recall)
        f1list.append(f1)
        W_list.append(np.sum(cm[:,k]))

        TP_sum += TP
        FP_sum += FP
        FN_sum += FN

    W_list = W_list/np.sum(W_list)

    #Micro
    Micro_pre = TP_sum/(TP_sum+FP_sum)
    Micro_recall = TP_sum/(TP_sum+FN_sum)
    Micro_F1 = 2*Micro_pre*Micro_recall/(Micro_pre + Micro_recall)

    #Macro
    Macro_pre =   np.mean(prelist)
    Macro_recall =   np.mean(recalllist)
    Macro_f1 =   np.mean(f1list)

    #average
    weighted_pre = np.sum(W_list * prelist)
    weighted_recall = np.sum(W_list * recalllist)
    weighted_f1 = 2*weighted_pre*weighted_recall/(weighted_pre + weighted_recall)


    return Macro_pre,Macro_recall,Macro_f1
    # return Micro_pre,Micro_recall,Micro_F1
    # return weighted_pre,weighted_recall,weighted_f1


def getRankedPredictedLabels(pre_List,GT_list):
    GT_list = np.array(GT_list)
    label_list = np.sort(np.unique(GT_list))

    recall_list = np.zeros(len(label_list))
    size_list = np.zeros(len(label_list))

    for n,pre in enumerate(pre_List):
        GT = GT_list[n]
        index = np.where(label_list==GT)
        size_list[index] += 1
        if(GT == pre[0]):
            recall_list[index] += 1
    recall_list = recall_list/size_list
    indexList = np.argsort(recall_list)

    worstList = label_list[indexList[:1]]
    bestlist = label_list[indexList[-1:]]

    return worstList,bestlist





if __name__=='__main__':

    mode = 1
    if mode == 0: #分析各种评价指标
        print('--------------------------------------------------------------------------------------------------')  
        print(f"{'No':<4}\t{'filename':<50}\t{'acc1':<8}{'acc5':<8}{'pre':<8}{'Recall':<8}{'F1':<8}")
        filePath = './results'
        fileList = os.listdir(filePath)

        for t,filename in enumerate(fileList):
            if not '.pkl' in filename:
                continue

            result_file = '/'.join([filePath,filename])
            with open(result_file,'rb') as f:
                results = pickle.load(f)


            #Top1 & Top5
            acc1 = 0
            acc5 = 0
            N_samples = len(results['pre'])
            for n,pre_label in enumerate(results['pre']):
                GT = results['GT'][n]
                if GT in pre_label: #TODO: text need to be further index
                    acc5 = acc5 + 1
                if GT == pre_label[0]:
                    acc1 = acc1 + 1
            acc_rate5 = acc5 / N_samples
            acc_rate1 = acc1 / N_samples

            #Macro
            cm = cal_confusion_matrix(results['pre'],results['GT'])
            pre,recall,F1 = evaluate(cm)
            
            # print(f"No.{n:<2} {filename:<35}:\tpre = {pre:.4f},Recall = {recall:.4f},F1 = {F1:.4f}")
            print(f"{t:<4}\t{filename:<50}\t{acc_rate1:.4f}\t{acc_rate5:.4f}\t{pre:.4f}\t{recall:.4f}\t{F1:.4f}")
        
        print('--------------------------------------------------------------------------------------------------')  
    elif mode == 1: #分析每种方法分错的类别排序
        print('--------------------------------------------------------------------------------------------------')  
        filePath = './results'
        fileList = os.listdir(filePath)

        for t,filename in enumerate(fileList):
            if not '.pkl' in filename:
                continue

            result_file = '/'.join([filePath,filename])
            with open(result_file,'rb') as f:
                results = pickle.load(f)

            if t in [6,10,15]:
                indexs = [n for n,GT in enumerate(results['GT']) if GT=='2518689634']
                pre = [results['pre'][n][0] for n in indexs]
                t = t

            worstList,bestlist  = getRankedPredictedLabels(results['pre'],results['GT'])
            print(f"{t:<4}\t{filename:<50}",worstList,bestlist)


        

