import numpy as np
import torch
import torch.nn as nn
import os.path as osp
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import os
import argparse
from lib.config.hrnet_config import update_config
from lib.config.hrnet_config import config
from lib.datasets.trip_dataloader import text_retrieval
from pathlib import Path
from lib.models.text_tfheader import TextEncoder
from lib.utils.loss import triplet_loss_cl
from tqdm import tqdm


class text_simple_tf(nn.Module):
    def __init__(self, original_dim, is_transform = True):
        super(text_simple_tf,self).__init__()
        # add one mlp to fuse the dimension
        self.transform = is_transform
        self.linear1 = nn.Sequential(nn.Linear(original_dim, 4096), nn.BatchNorm1d(4096),nn.LeakyReLU(),
                                        nn.Linear(4096,1024), nn.BatchNorm1d(1024), nn.LeakyReLU())
        self.linear2 = nn.Sequential(nn.Linear(original_dim,1024), nn.BatchNorm1d(1024), nn.LeakyReLU())

        self.textencoder = TextEncoder()
    def forward(self,feature):
        batch_size = feature.shape[0]
        if self.transform:
            fusion_feature = torch.cat([self.linear1(feature.squeeze(-1)).unsqueeze(-1) , self.linear2(feature.squeeze(-1)).unsqueeze(-1)], dim=-1)
            embeded_feature = self.textencoder(fusion_feature)
            extracted_feature = nn.functional.adaptive_max_pool1d(embeded_feature,1) 
            extracted_feature = extracted_feature.reshape(batch_size,-1)
            output_feature = extracted_feature / torch.norm(extracted_feature,dim=-1,keepdim=True)
        else:
            embeded_feature = self.linear1(feature.squeeze(-1)) + self.linear2(feature.squeeze(-1))
            output_feature = embeded_feature / torch.norm(embeded_feature,dim=-1,keepdim=True)

        return output_feature

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg) # 把config的文件更新过去
    return args

def get_optimizer(model):
    lr = config.TRAIN.LR
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr) # 整体模型权重均全部重新训练
    return model, optimizer

def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    # import pdb;pdb.set_trace()
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        metrics = checkpoint['loss']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, metrics

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, np.inf


def main():
    args = parse_args() # 读取 cfg 参数，config表示之后需要看一下

    gpus = [int(i) for i in config.GPUS.split(',')]
    image_folder = '/Extra/panzhiyu/img_retrieval/shopee-product-matching/train_images'
    back_dataset = text_retrieval(image_folder,is_train = True)
    test_dataset = text_retrieval(image_folder,is_train = False)
    back_loader = torch.utils.data.DataLoader(
        back_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    print('=> Constructing models ..')
    model = text_simple_tf(original_dim = 11914, is_transform=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model, optimizer = get_optimizer(model)
    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    least_test_loss = np.inf # enough large

    # if config.NETWORK.PRETRAINED_BACKBONE: # no pretrained test   
    #     print(f'Using backbone {config.NETWORK.PRETRAINED_BACKBONE}')
    #     model = load_backbone(model, config.NETWORK.PRETRAINED_BACKBONE) # load POSE ESTIMATION BACKBONE
    
    # if config.TRAIN.RESUME:
    #     start_epoch, model, optimizer, metrics_load = load_checkpoint(model, optimizer, config.OUTPUT_DIR) # TODO: Load the A1 metrics
    best_model = torch.load(os.path.join(config.OUTPUT_DIR ,config.TEST.MODEL_FILE))

    model.module.load_state_dict(best_model)
    
    print('=> EVAL...')
    device=torch.device('cuda')
    model.eval()
    # construct the backend lib
    back_feature = []
    back_label = []
    for i,batch in tqdm(enumerate(back_loader)):
        image, label = batch
        image = image.to(device)
        features = model(image)
        back_feature.append(features.detach())
        back_label.extend(label)
    back_feature = torch.cat(back_feature,dim=0)
    back_label = np.array(back_label)
    
    # testing
    acc1 = 0
    acc5 = 0
    for j, batch in tqdm(enumerate(test_loader)):
        test_image, test_label = batch
        test_image = test_image.to(device)
        test_features = model(test_image)
        test_num = test_image.shape[0]
        for t in range(test_num):
            quary_feature = test_features[t].reshape(-1,1)
            inner_product = back_feature @ quary_feature
            inner_product = inner_product.reshape(-1)
            _ , index = torch.sort(inner_product)
            # import pdb;pdb.set_trace()
            index = torch.flip(index,dims=[0]) 
            index_5 = index[:5]
            pred_label = back_label[index_5.cpu()]
            if test_label[t] in pred_label:
                acc5 = acc5 + 1
            if test_label[t] == pred_label[0]:
                acc1 = acc1 + 1

    acc_rate5 = acc5 / len(test_dataset)
    acc_rate1 = acc1 / len(test_dataset)
    print('----------------------------')

    print(f"acc1 = {acc_rate1:.4f}")
    print(f"acc5 = {acc_rate5:.4f}")
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
