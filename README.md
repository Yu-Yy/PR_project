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
| nmf     | 0.4275    | 0.5175     | xxx |
| pca     | 0.5650    | 0.6025     | xxx |
| pca_aug     | 0.5725     | 0.5950     | xxx |
| SIFT+KNN     | 0.6587    | 0.7460     | xxx |
| HRnet_base     | 0.6125    | 0.7200     | xxx |
| HRnet+Transformer     | 0.5725     | 0.6225     | xxx |
| ResNet |  0.8450   |  0.9400   |  xx   |



### text:
- Result

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| Text_pure     | 0.8400     | 0.9025     |
| NMF     | 0.6950    | 0.7975     |
| PCA     | 0.7500    | 0.8475     |
| MLP     | 0.9025     | 0.9550     |
| MLP+Transformer     | 0.9175    | 0.9600     |

### text + image

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| hrnet_Transformer_Fusion     | 0.6325     | 0.7650     |
| HRnet + Text_Trans     | 0.6250  |   0.7375   |
| Resnet + Text_Trans     |  0.7725   | 0.8700  |
| Borda_method (res + Text_Trans)     | 0.9575     | 0.9950  |


## Model download
You can download the model `result` [here](https://cloud.tsinghua.edu.cn/d/42a31128af9d401f8aa9/)

## Data download
You can download the dataset of `shopee-product-matching/` by this [link](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/) 

## Data preprocess
In our experiment, we divide the original dataset into training set, test set for training the deep nerual network. Besides, we 
extract a mini dataset for evaluate different methods. The evaluation method is that using the test set of the mini one as quiries and trainning set of that as gallaries.
The cooresponding code is in `utils/dataset_preprocess`.<br>
