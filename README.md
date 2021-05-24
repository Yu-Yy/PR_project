# PR_project
2020-2021 Partern Recognition for image retrieval

## Current Result
Mini_dataset: including 500 labels

### image:
pca:   acc1: 0.5605  acc5: 0.5932

pcaaug:  acc1: 0.5733 acc5: 0.5917

SIFT + KNN: acc1: 0.6587 acc5: 0.7460

Naive DL：epoch 4 acc1: 0.4794, acc5: 0.5789
          epoch 30 acc1: 0.6060 acc5: 0.6970
          epoch 50 acc1: 0.5946 acc5: 0.7055

Transformer DL: epoch 11 acc1 = 0.5761 acc5 = 0.6188
                epoch 30 acc1 = 0.5377 acc5 = 0.5647


### text:
text_pure:  acc: 0.8492 acc5: 0.9132

nmf: acc: 0.6785 acc5: 0.8137

pca:  acc: 0.7297 acc5: 0.8236

mlp: acc1: 0.8848 acc5: 0.9445

mlp + Transformer: acc1: 0.9090 acc5: 0.9630

### text + image
Transformer_Fusion: acc1: 0.6714 acc5: 0.8009 