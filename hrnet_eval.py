import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter
import sys
import os
import argparse
from lib.config.hrnet_config import update_config
from lib.config.hrnet_config import config
from lib.models.Hrnet import HigherResolutionNet
from lib.datasets.trip_dataloader import val_retrieval
from pathlib import Path
# loss
from lib.utils.loss import triplet_loss_cl
from tqdm import tqdm
from lib.models.Vit_header import VitEncoder


class retrieval_net(nn.Module):
    def __init__(self,cfg, is_train = True, is_transform=True):
        super(retrieval_net,self).__init__()
        self.backbone = HigherResolutionNet(cfg, is_train=is_train)
        self.is_transform = is_transform
        if self.is_transform:
            self.self_attention = VitEncoder(cfg.MODEL_EXTRA.STAGE4.NUM_CHANNELS[0])
    def forward(self, images):
        features = self.backbone(images)
        batch_size = features.shape[0]
        if self.is_transform:
            output_feature = self.self_attention(features)
        else:
            extracted_feature = nn.functional.adaptive_avg_pool2d(features,(1,1))
            output_feature = extracted_feature.reshape(batch_size,-1)
        # need to normalize the feature
        output_feature = output_feature / torch.norm(output_feature,dim=-1,keepdim=True)
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

def load_backbone(model,pretrained_file):
    pretrained_state_dict = torch.load(pretrained_file)
    model_state_dict_backbone = model.module.backbone.state_dict()
    prefix_b = 'backbone.'
    new_pretrained_state_dict_bacbone = {}
    for k, v in pretrained_state_dict.items():
        if k.replace(prefix_b, "") in model_state_dict_backbone and v.shape == model_state_dict_backbone[k.replace(prefix_b, "")].shape:     #.replace(prefix, "") .replace(prefix, "")
            new_pretrained_state_dict_bacbone[k.replace(prefix_b, "")] = v
    print("load statedict from {}".format(pretrained_file))
    model.module.backbone.load_state_dict(new_pretrained_state_dict_bacbone)
    return model

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

def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

def main():
    args = parse_args() # 读取 cfg 参数，config表示之后需要看一下

    gpus = [int(i) for i in config.GPUS.split(',')]
    image_folder = '/Extra/panzhiyu/img_retrieval/shopee-product-matching/train_images'
    back_dataset = val_retrieval(image_folder,is_train = True)
    test_dataset = val_retrieval(image_folder,is_train = False)
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
    model = retrieval_net(config, is_train= True, is_transform=True)
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
