import torch
import numpy as np


class Negative_loss(torch.nn.Module):
    def __init__(self):
        super(Negative_loss, self).__init__()

    def forward(self, output, pred_avg):
        loss2 = 0
        for i, cur_output in enumerate(output):
            if i == 0:
                output_del = output[1:, :]
                a = (cur_output - pred_avg)
                b = torch.sum((output_del - pred_avg), dim=0)
                c = torch.mul(a, b)
                molecule = torch.sum(c)
                loss2 += molecule / len(pred_avg)
            else:
                output_del = torch.cat((output[0:i, :], output[i + 1:, :]), dim=0)
                a = (cur_output - pred_avg)
                b = torch.sum((output_del - pred_avg), dim=0)
                c = torch.mul(a, b)
                molecule = torch.sum(c)
                loss2 += molecule / len(pred_avg)

        return loss2
