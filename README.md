# PR_project
2020-2021 Partern Recognition for image retrieval: you can refer to our project's codes via our [github repository](https://github.com/Yu-Yy/PR_project) 

## Basic requirement
The codes are running under:
```
numpy
pytorch
torchvision
scikit-learn
scikit-image
opencv-python
tensorboardX
pickle
csv
```

## Result
Mini_dataset: including 500 labels

### image:
- Result

| Method     | acc1     | acc5     | recall     |
| ---------- | :-----------:  | :-----------: | :-----------: |
| nmf     | 0.3698    | 0.4481     | xxx |
| pca     | 0.5605     | 0.5932     | xxx |
| pca_aug     | 0.5733     | 0.5917     | xxx |
| SIFT+KNN     | 0.6587    | 0.7460     | xxx |
| HRnet_base     | 0.6060     | 0.6970     | xxx |
| HRnet+Transformer     | 0.5818     | 0.6230     | xxx |
| ResNet |xx |xx |xx |


### text:
- Result

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| Text_pure     | 0.8492     | 0.9132     |
| NMF     | 0.6785     | 0.8137     |
| PCA     | 0.7297    | 0.8236     |
| MLP     | 0.8848     | 0.9445     |
| MLP+Transformer     | 0.9132    | 0.9602     |

### text + image

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| Transformer_Fusion     | 0.6714     | 0.8009     |
| HRnet + Text_Trans     |      |      |
| Voting_method     |      |      |


## Model download
You can download the model `result` [here](https://cloud.tsinghua.edu.cn/d/42a31128af9d401f8aa9/)

## Data download
You can download the dataset of `shopee-product-matching/` by this [link](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/) 

## Data preprocess
In our experiment, we divide the original dataset into training set, test set for training the deep nerual network. Besides, we 
extract a mini dataset for evaluate different methods. The evaluation method is that using the test set of the mini one as quiries and trainning set of that as gallaries.
The cooresponding code is in `utils/dataset_preprocess`.<br>
