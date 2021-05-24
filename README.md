# PR_project
2020-2021 Partern Recognition for image retrieval

## Current Result
Mini_dataset: including 500 labels

## Model download
You can download the model `result` [here](https://cloud.tsinghua.edu.cn/d/42a31128af9d401f8aa9/)

### image:
- Result

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| pca     | 0.5605     | 0.5932     |
| pca_aug     | 0.5733     | 0.5917     |
| SIFT+KNN     | 0.6587    | 0.7460     |
| HRnet_base     | 0.6060     | 0.6970     |
| HRnet+Transformer     | 0.5761     | 0.6188     |


### text:
- Result

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| Text_pure     | 0.8492     | 0.9132     |gi
| NMF     | 0.6785     | 0.8137     |
| PCA     | 0.7297    | 0.8236     |
| MLP     | 0.8848     | 0.9445     |
| MLP+Transformer     | 0.9090     | 0.9630     |

### text + image

| Method     | acc1     | acc5     |
| ---------- | :-----------:  | :-----------: |
| Transformer_Fusion     | 0.6714     | 0.8009     |