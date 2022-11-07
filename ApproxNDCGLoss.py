import torch
import numpy as np


def torch_dcg_at_k(batch_sorted_labels, cutoff=None):
    if cutoff is None:
        cutoff = batch_sorted_labels.size(1)
    batch_numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
    batch_discounts = torch.log2(torch.arange(cutoff).type(torch.cuda.FloatTensor).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators / batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k


# 计算Rank位置的近似
def get_approx_ranks(input, alpha=10):
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)
    batch_indicators = torch.sigmoid(alpha * torch.transpose(batch_pred_diffs, dim0=1, dim1=2))
    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5
    return batch_hat_pis


def approxNDCG_loss(batch_preds=None, batch_stds=None, alpha=10):
    # 计算Rank位置的近似
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha)
    # 计算理想情况的DCG
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None)
    # 计算预测情况的DCG
    batch_gains = torch.pow(2.0, batch_stds) - 1.0

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1).unsqueeze(dim=1)
    # 计算NDCG
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)
    batch_loss = 1 - torch.mean(batch_approx_nDCG)
    return batch_loss


class ApproxNDCG(torch.nn.Module):
    def __init__(self, model_para_dict=10):
        super(ApproxNDCG, self).__init__()
        self.alpha = model_para_dict

    def getCorrel(self, labels, Max_index, Min_index):
        y_true = torch.zeros(2, len(labels) - 1).cuda()
        # 按最大值进行排列
        if Max_index == 0:
            labels_del = labels[1:].squeeze()
            y_true[0, :] = labels[Max_index] - labels_del
        else:
            labels_del = torch.cat((labels[0:Max_index], labels[Max_index + 1:]), dim=0).squeeze()
            y_true[0, :] = labels[Max_index] - labels_del
        # 按最小值进行排列
        if Min_index == 0:
            labels_del = labels[1:].squeeze()
            y_true[1, :] = labels_del - labels[Min_index]
        else:
            labels_del = torch.cat((labels[0:Min_index], labels[Min_index + 1:]), dim=0).squeeze()
            y_true[1, :] = labels_del - labels[Min_index]
        return y_true

    def getSimilar(self, pred, Max_index, Min_index):
        y_pred = torch.zeros(2, len(pred) - 1).cuda()
        # 按最大值进行排列
        if Max_index == 0:
            labels_del = pred[1:].squeeze()
            y_pred[0, :] = pred[Max_index] - labels_del
        else:
            labels_del = torch.cat((pred[0:Max_index], pred[Max_index + 1:]), dim=0).squeeze()
            y_pred[0, :] = pred[Max_index] - labels_del
        # 按最小值进行排列
        if Min_index == 0:
            labels_del = pred[1:].squeeze()
            y_pred[1, :] = labels_del - pred[Min_index]
        else:
            labels_del = torch.cat((pred[0:Min_index], pred[Min_index + 1:]), dim=0).squeeze()
            y_pred[1, :] = labels_del - pred[Min_index]

        return y_pred

    def forward(self, pred, labels, Max_index, Min_index, max_target, min_target, T):
        batch_stds = self.getCorrel(labels, Max_index, Min_index)
        batch_stds = 1 - (batch_stds / (max_target - min_target))
        batch_stds = batch_stds * T
        # batch_stds_numpy = batch_stds.cpu().numpy()

        batch_preds = self.getSimilar(pred, Max_index, Min_index)
        batch_preds = 1 - batch_preds

        target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
        target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)

        batch_loss = approxNDCG_loss(target_batch_preds, target_batch_stds, self.alpha)

        return batch_loss
