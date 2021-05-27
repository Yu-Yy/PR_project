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

| Method     | Acc1     | Acc5     | Precision     | Recall | F1 |
| ---------- | :-----------:  | :-----------: | :-----------: |:-----------: |:-----------: |
| nmf     | 0.4275    | 0.5175     | 0.3048 | 0.2571    |  0.2640    |
| pca     | 0.5650    | 0.6025     | 0.4429 | 0.3862    |  0.3930      |
| pca_aug     | 0.5725     | 0.5950     | 0.4444 | 0.3822 | 0.3970  |
| SIFT+KNN     | 0.6587    | 0.7460     | 0.5319 | 0.4765  | 0.4869  |
| HRnet_base     | 0.6125    | 0.7200     | 0.4396 | 0.3897  | 0.3987  |
| HRnet+Transformer     | 0.5725     | 0.6225     | 0.4297 | 0.3708  | 0.3858  |
| ResNet |  0.8450   |  0.9400   |  0.7019  | 0.6708  | 0.6726  |



### text:
- Result

| Method     | Acc1     | Acc5     | Precision    |  Recall   |  F1  |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
| Text_pure     | 0.8400     | 0.9025     | 0.8001 | 0.7271 | 0.7502 |
| NMF     | 0.6950    | 0.7975     | 0.4629  | 0.4024 | 0.4160 |
| PCA     | 0.7500    | 0.8475     |  |  |  |
| MLP     | 0.9025     | 0.9550     | 0.8021 | 0.7685 | 0.7752 |
| MLP+Transformer     | 0.9175    | 0.9600     | 0.8206 | 0.8024 | 0.8044 |

### text + image

| Method     | Acc1     | Acc5     | Precision  |  Recall  | F1  |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
| hrnet_Transformer_Fusion     | 0.6325     | 0.7650     | 0.4352 | 0.3947 | 0.3995 |
| HRnet + Text_Trans     | 0.6250  |   0.7375   | 0.4675 | 0.4268 | 0.4267 |
| Resnet + Text_Trans     |  0.7725   | 0.8700  | 0.6607 | 0.6089 | 0.6175 |
| Borda_method (res + Text_Trans)     | 0.9575     | 0.9950  | 0.9228 | 0.8739 | 0.8882  |


## Model download
You can download the model `result` [here](https://cloud.tsinghua.edu.cn/d/42a31128af9d401f8aa9/)

## Data download
You can download the dataset of `shopee-product-matching/` by this [link](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/) 

## Data preprocess
In our experiment, we divide the original dataset into training set, test set for training the deep nerual network. Besides, we 
extract a mini dataset for evaluate different methods. The evaluation method is that using the test set of the mini one as quiries and trainning set of that as gallaries.
The cooresponding code is in `utils/dataset_preprocess`.<br>
