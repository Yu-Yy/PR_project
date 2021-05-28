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
In order to install the gist package, please refer to this [link](https://github.com/Kalafinaian/python-img_gist_feature)

## Result & Corresponding code
Mini_dataset: including 500 labels

### image:

- Corresponding Codes
`test_pca.py` for testing image pca, `test_nmf.py` for testing image nmf.<br>
`hrnet_retrieval.py` for HRNet image train and `hrnet_eval.py` for HRNet model test.<br>
`resnet_retrieval.py` for ResNet image train and `resnet_eval.py` for ResNet image test.<br>

- Result

| Method     | Acc1     | Acc5     | Precision     | Recall | F1 |
| ---------- | :-----------:  | :-----------: | :-----------: |:-----------: |:-----------: |
| NMF     | 0.4275    | 0.5175     | 0.3048 | 0.2571    |  0.2640    |
| PCA     | 0.5650    | 0.6025     | 0.4429 | 0.3862    |  0.3930      |
| PCA(data augmentation)     | 0.5725     | 0.5950     | 0.4444 | 0.3822 | 0.3970  |
| SIFT+KNN     | 0.6375    | 0.7025     | 0.5319 | 0.4765  | 0.4869  |
| HRnet     | 0.6125    | 0.7200     | 0.4396 | 0.3897  | 0.3987  |
| HRnet+Transformer     | 0.5725     | 0.6225     | 0.4297 | 0.3708  | 0.3858  |
| ResNet |  0.8450   |  0.9400   |  0.7019  | 0.6708  | 0.6726  |


### text:
- Corresponding Codes
`text_pure.py` for pure (initial) text feature, PCA, NMF testing.<br>
`dl_text.py` for text MLP (transformer) train, and `dl_text_eval.py` for text MLP (transformer) test.<br>

- Result

| Method     | Acc1     | Acc5     | Precision    |  Recall   |  F1  |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
| Text_pure(log(freq))     | 0.8400     | 0.9025     | 0.8001 | 0.7271 | 0.7502 |
| Text_pure(tf_idf)     | 0.8650     | 0.9725     | 0.8702 | 0.7929 | 0.8195 |
| NMF     | 0.6950    | 0.7975     | 0.4629  | 0.4024 | 0.4160 |
| PCA     | 0.7550    | 0.8550     | 0.5552  | 0.4832 | 0.5027 |
| MLP     | 0.9025     | 0.9550     | 0.8021 | 0.7685 | 0.7752 |
| MLP+Transformer     | 0.9175    | 0.9600     | 0.8206 | 0.8024 | 0.8044 |

### text + image
- Corresponding Codes
`image_text_fusion.py` for hrnet (transformer) and text transformer model fusion train, and `fusion_eval.py` for testing.
`resnet_text_fusion.py` for resnet and text transformer model fusion train, and `resnet_text_fuse_eval` for testing.


- Result

| Method     | Acc1     | Acc5     | Precision  |  Recall  | F1  |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
| Image(hrnet+Transformer) + Text(MLP+Transformer) | 0.6325     | 0.7650     | 0.4352 | 0.3947 | 0.3995 |
| Image(HRnet) + Text(MLP+Transformer)     | 0.6250  |   0.7375   | 0.4675 | 0.4268 | 0.4267 |
| Image(Resnet) + Text(MLP+Transformer)     |  0.7950   | 0.8850  | 0.6572 | 0.6065 | 0.6154 |
| Borda_method (res + Text_Trans)     | 0.9575     | 0.9950  | 0.9228 | 0.8739 | 0.8882  |

## Model download
You can download the model `result` [here](https://cloud.tsinghua.edu.cn/d/42a31128af9d401f8aa9/), and put in this directory.

## Data download
You can download the dataset of `shopee-product-matching/` by this [link](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/)  and uncompress, put in this directory.

## Data preprocess
In our experiment, we divide the original dataset into training set, test set for training the deep nerual network. Besides, we 
extract a mini dataset (including 500 labels) for evaluate different methods. The evaluation method is that using the test set of the mini one as quiries and trainning set of that as gallaries. The divided datasets (for deep neural network trainning) file name is in `datasets/train_divide_eq.pkl`, `datasets/test_divide_eq.pkl`. Mini datasets file name is in `datasets/train_divide_mini500_eq.pkl`, `datasets/val_divide_mini500_eq.pkl`, `datasets/test_divide_mini500_eq.pkl`. <br>
The cooresponding processing code is in `utils/dataset_preprocess.py`. <br>
The code for data augmentation is in `utils/data_augmentation.py`, the code for generate tbe Bag of Words is in `utils/create_BOW.py`.<br>

## Code Block Illustration
`configs` contains yaml files about the configurations of network running codes.<br>
`lib` contains the main structured file of network running codes.<br>
     `lib/config` contains the basic hrnet configurations. <br>
     `lib/datasets` contains the dataset class of neural networks. <br>
     `lib/model` contains the model structure of neural networks. <br>
     `lib/untils` contains the loss functions of neural networks. <br>
`utils` is about the data preprocess.<br>

## Running code
### image
- traditional methods
```
python test_pca --aug True
python test_nmf --aug True
```
- neural networks
```
# Train the hrnet without transformer 
CUDA_VISIBLE_DEVICES=0,1 python hrnet_retrieval.py --cfg configs/hrnet.yaml --transform False
# Test the hrnet   
CUDA_VISIBLE_DEVICES=0,1 python hrnet_eval.py --cfg configs/hrnet.yaml --transform False 
# Train the resnet
CUDA_VISIBLE_DEVICES=0,1 python resnet_retrieval.py --cfg configs/resnet.yaml
# Test the resnet
CUDA_VISIBLE_DEVICES=0,1 python resnet_eval.py --cfg configs/resnet.yaml
```
### Text
- traditional methods
```
# change the code inside for selecting pca or nmf
python text_pure
```
- neural networks
```
# Train the mlp
CUDA_VISIBLE_DEVICES=0,1 python dl_text.py --cfg configs/text.yaml --transform False
# Test the mlp
CUDA_VISIBLE_DEVICES=0,1 python dl_text_eval.py --cfg configs/text.yaml --transform False
# Train the mlp + transformer
CUDA_VISIBLE_DEVICES=0,1 python dl_text.py --cfg configs/text.yaml --transform True
# Test the mlp + transformer 
CUDA_VISIBLE_DEVICES=0,1 python dl_text_eval.py --cfg configs/text.yaml --transform True
```

## cross modal
- neural networks
```
# Train the hrnet + text&transformer
CUDA_VISIBLE_DEVICES=0,1 python image_text_fusion.py --cfg configs/fusing.yaml
# Test the hrnet + text&transformer
CUDA_VISIBLE_DEVICES=0,1 python fusion_eval.py --cfg configs/fusing.yaml
# Train the resnet + text&transformer
CUDA_VISIBLE_DEVICES=0,1 python resnet_text_fusion.py --cfg configs/fusing.yaml
# Test the resnet + text&transformer
CUDA_VISIBLE_DEVICES=0,1 python resnet_text_fuse_eval.py --cfg configs/fusing.yaml
```