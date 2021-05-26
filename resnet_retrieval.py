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
# from lib.models.Hrnet import HigherResolutionNet
import torchvision
from lib.datasets.trip_dataloader import trip_retrieval
from pathlib import Path
# loss
from lib.utils.loss import triplet_loss_cl
# from lib.models.Vit_header import VitEncoder

class retrieval_net(nn.Module):
    def __init__(self,cfg, is_train = True): #, is_transform=True
        super(retrieval_net,self).__init__()
        # self.backbone = HigherResolutionNet(cfg, is_train=is_train)
        self.backbone = torchvision.models.resnet50(pretrained=True, progress=True)
        # self.is_transform = is_transform
        # if self.is_transform:
        #     self.self_attention = VitEncoder(cfg.MODEL_EXTRA.STAGE4.NUM_CHANNELS[0])
    def forward(self, images):
        features = self.backbone(images) # 1000 dim feature output
        # import pdb; pdb.set_trace()
        # batch_size = features.shape[0]
        # if self.is_transform:
        #     output_feature = self.self_attention(features)
        # else:
        #     extracted_feature = nn.functional.adaptive_avg_pool2d(features,(1,1))
        #     output_feature = extracted_feature.reshape(batch_size,-1)
        # need to normalize the feature
        output_feature = features / torch.norm(features,dim=-1,keepdim=True)
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
    result_log_dir = Path(config.OUTPUT_DIR)
    result_log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(i) for i in config.GPUS.split(',')]
    image_folder = '/Extra/panzhiyu/img_retrieval/shopee-product-matching/train_images'
    train_dataset = trip_retrieval(image_folder,is_train = True)
    test_dataset = trip_retrieval(image_folder,is_train = False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
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
    model = retrieval_net(config, is_train= True) #, is_transform=False
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model, optimizer = get_optimizer(model)
    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    least_test_loss = np.inf # enough large

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, metrics_load = load_checkpoint(model, optimizer, config.OUTPUT_DIR) # TODO: Load the A1 metrics
        least_test_loss = metrics_load

    tb_log_dir = Path(os.path.join(config.OUTPUT_DIR,'tensorboard_log'))
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    writer_dict = {
        'writer': SummaryWriter(log_dir=str(tb_log_dir)),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    
    print('=> Training...')
    device=torch.device('cuda')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))
        train_sim_loss = AverageMeter()
        test_sim_loss = AverageMeter()
        trip_class_loss = triplet_loss_cl()
        # The train part 
        model.train()
        for i, batch in enumerate(train_loader):
            # import pdb; pdb.set_trace()
            q_images, g_images = batch
            if q_images.shape[0] == 1:
                continue # cannot do the triplet loss 
            q_images = q_images.to(device)
            g_images = g_images.to(device)

            q_features = model(q_images)
            g_features = model(g_images)

            # calculate the loss as triplet
            trip_loss = trip_class_loss(q_features,g_features)
            train_sim_loss.update(trip_loss.item())
            optimizer.zero_grad()
            trip_loss.backward()
            optimizer.step()

            if i % config.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = f'Epoch:[{epoch}][{i}/{len(train_loader)}]\t'\
                        f'Loss_trip: {train_sim_loss.val:.3f}({train_sim_loss.avg:.3f})\t'\
                        f'Memory {gpu_memory_usage:.1f}'
                print(msg)
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss_trip', train_sim_loss.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        # store the first model
        if epoch ==  0:
            model_name =os.path.join(config.OUTPUT_DIR,
                                          f'epoch{epoch}_state.pth.tar')
            print('saving current model state to {}'.format(
                model_name))
            torch.save(model.module.state_dict(), model_name)
        # The eval part
        model.eval()
        for i, batch in enumerate(test_loader):
            q_images, g_images = batch
            if q_images.shape[0] == 1:
                continue # cannot do the triplet loss 
            q_images = q_images.to(device)
            g_images = g_images.to(device)

            q_features = model(q_images)
            g_features = model(g_images)

            # calculate the loss as triplet
            trip_loss = trip_class_loss(q_features,g_features)
            test_sim_loss.update(trip_loss.item())

            if i % config.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = f'Test:[{epoch}][{i}/{len(test_loader)}]\t'\
                        f'Loss_trip: {test_sim_loss.val:.3f}({test_sim_loss.avg:.3f})\t'\
                        f'Memory {gpu_memory_usage:.1f}'
                print(msg)
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar('test_loss_trip', test_sim_loss.avg, global_steps)
                writer_dict['test_global_steps'] = global_steps + 1
                

        test_loss = test_sim_loss.avg
        # compare the loss
        if test_loss < least_test_loss:
            least_test_loss = test_loss
            best_model = True
        else:
            best_model = False
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'loss': test_loss,
            'optimizer': optimizer.state_dict(),
        }, best_model, config.OUTPUT_DIR)
        
    final_model_state_file = os.path.join(config.OUTPUT_DIR,
                                          'final_state.pth.tar')
    torch.save(model.module.state_dict(), final_model_state_file)

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
