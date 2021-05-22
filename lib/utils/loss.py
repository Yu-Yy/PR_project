import torch
import torch.nn as nn
import torch.nn.functional as F

class triplet_loss_cl(nn.Module):
    def __init__(self):
        super(triplet_loss_cl,self).__init__()
    def forward(self, q_features, g_features):
        compare_num = q_features.shape[0]
        loss_ = 0
        for i in range(compare_num):
            quary = q_features[i].reshape(-1,1)
            inner_product = g_features @ quary
            exp_product = torch.exp(inner_product)
            exp_sum = torch.sum(exp_product)
            # import pdb;pdb.set_trace()
            log_term = - torch.log((exp_product[i]/exp_sum) + 1e-5) # avoid nan
            if torch.isnan(log_term):
                import pdb;pdb.set_trace()

            loss_ = loss_ + log_term
        
        return loss_ / compare_num

