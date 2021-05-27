import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import os
import argparse
from lib.config.hrnet_config import update_config
from lib.config.hrnet_config import config
from lib.datasets.trip_dataloader import image_text_eval
from pathlib import Path
from lib.utils.loss import triplet_loss_cl
# import the model
from resnet_retrieval import retrieval_net
from dl_text import text_simple_tf
from lib.models.text_tfheader import TextEncoder
import pickle
from tqdm import tqdm

class cross_modal(nn.Module):
    def __init__(self,cfg, original_dim, is_train = True, is_transform = True):
        super(cross_modal,self).__init__()
        self.image_em = retrieval_net(cfg, is_train = is_train, is_transform = False) #
        self.text_em = text_simple_tf(original_dim,is_transform)
        self.is_transformf = False
        if self.is_transformf:
            self.fusing_tr = TextEncoder()
            self.tr_layer = nn.Sequential(nn.Linear(1024, 1000), nn.BatchNorm1d(1000), nn.LeakyReLU())
        self.Linear_fusing1 = nn.Sequential(nn.Linear(1024 + 1000, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU())
        self.Linear_fusing2 = nn.Sequential(nn.Linear(1024 + 1000, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(), nn.Linear(2048,1024), nn.BatchNorm1d(1024),nn.LeakyReLU())
    def forward(self,image, text_feature):
        image_feature = self.image_em(image)
        text_embed = self.text_em(text_feature)
        # import pdb;pdb.set_trace()
        if not self.is_transformf:
            Fusion_f = torch.cat([image_feature, text_embed], dim=-1)
            Fusion_f = self.Linear_fusing1(Fusion_f) + self.Linear_fusing2(Fusion_f)
        else:
            tr_text = self.tr_layer(text_embed)
            fuse = torch.cat([image_feature.unsqueeze(-1), tr_text.unsqueeze(-1)],dim=-1)
            fusing = self.fusing_tr(fuse)
            fusing = nn.functional.adaptive_max_pool1d(fusing ,1)
            Fusion_f = fusing.squeeze(-1)

        output_feature = Fusion_f / torch.norm(Fusion_f,dim=-1,keepdim=True)
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

def main():
    args = parse_args() # 读取 cfg 参数，config表示之后需要看一下
    result_log_dir = Path(config.OUTPUT_DIR)
    result_log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(i) for i in config.GPUS.split(',')]
    image_folder = '/Extra/panzhiyu/img_retrieval/shopee-product-matching/train_images'
    train_dataset = image_text_eval(image_folder,is_train = True)
    test_dataset = image_text_eval(image_folder,is_train = False)
    back_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    print('=> Constructing models ..')
    model = cross_modal(config, original_dim=11914, is_transform=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model, optimizer = get_optimizer(model)
    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    least_test_loss = np.inf # enough large
    # pretrain_file_image = '/home/panzhiyu/Homework/img_retrieval/PR_project/result/pure_resnet/model_best.pth.tar'
    # pretrain_file_text = '/home/panzhiyu/Homework/img_retrieval/PR_project/result/trans_text/model_best.pth.tar'
    # if config.NETWORK.PRETRAINED_BACKBONE: # no pretrained test
    #     # load the pretrained two model
    #     pretrained_state_dict_image = torch.load(pretrain_file_image)
    #     pretrained_state_dict_text = torch.load(pretrain_file_text)
    #     model_state_dict_image = model.module.image_em.state_dict()
    #     model_state_dict_text = model.module.text_em.state_dict()

    #     prefix = ''# module.
    #     new_pretrained_state_dict_image = {}
    #     for k, v in pretrained_state_dict_image.items():
    #         if k.replace(prefix, "") in model_state_dict_image and v.shape == model_state_dict_image[k.replace(prefix, "")].shape:     #.replace(prefix, "") .replace(prefix, "")
    #             new_pretrained_state_dict_image[k.replace(prefix, "")] = v
    #     new_pretrained_state_dict_text = {}
    #     for k, v in pretrained_state_dict_text.items():
    #         if k.replace(prefix, "") in model_state_dict_text and v.shape == model_state_dict_text[k.replace(prefix, "")].shape:     #.replace(prefix, "") .replace(prefix, "")
    #             new_pretrained_state_dict_text[k.replace(prefix, "")] = v
    #     model.module.image_em.load_state_dict(new_pretrained_state_dict_image)
    #     model.module.text_em.load_state_dict(new_pretrained_state_dict_text)
    #     print('load backbone')
    #     # print(f'Using backbone {config.NETWORK.PRETRAINED_BACKBONE}')
    #     # model = load_backbone(model, config.NETWORK.PRETRAINED_BACKBONE) # load POSE ESTIMATION BACKBONE

    # if config.TRAIN.RESUME:
    #     start_epoch, model, optimizer, metrics_load = load_checkpoint(model, optimizer, config.OUTPUT_DIR) # TODO: Load the A1 metrics
    #     least_test_loss = metrics_load

    # tb_log_dir = Path(os.path.join(config.OUTPUT_DIR,'tensorboard_log'))
    # tb_log_dir.mkdir(parents=True, exist_ok=True)

    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=str(tb_log_dir)),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }
    best_model = torch.load(os.path.join(config.OUTPUT_DIR ,config.TEST.MODEL_FILE))

    model.module.load_state_dict(best_model)
    
    print('=> EVAL...')
    device=torch.device('cuda')
    model.eval()
    # construct the backend lib
    back_feature = []
    back_label = []
    for i,batch in tqdm(enumerate(back_loader)):
        image, text_feature,label = batch
        image = image.to(device)
        text_feature = text_feature.to(device)

        features = model(image, text_feature)
        back_feature.append(features.detach())
        back_label.extend(label)
    back_feature = torch.cat(back_feature,dim=0)
    back_label = np.array(back_label)

    acc1 = 0
    acc5 = 0
    results_image = {'pre':[],'GT':[]}
    for j, batch in tqdm(enumerate(test_loader)):
        test_image, test_text_feature ,test_label = batch
        test_image = test_image.to(device)
        test_text_feature = test_text_feature.to(device)
        test_features = model(test_image, test_text_feature)
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
            results_image['pre'].append(pred_label)
            results_image['GT'].append(test_label[t])
            if test_label[t] in pred_label:
                acc5 = acc5 + 1
            if test_label[t] == pred_label[0]:
                acc1 = acc1 + 1

    file_name = 'pred_result/res_tr_fusing_result.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(results_image , f)

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


