import torch
import numpy as np


class Ambiguity_loss(torch.nn.Module):
    def __init__(self):
        super(Ambiguity_loss, self).__init__()

    def forward(self, output, pred_avg):
        """
        Compute ambiguity_loss in a min-batch
        
        Arguments:
            output: The output of linear transformation(LT) layers in a min-batch
            pred_avg: The average value of output of all learners

        Returns:
            ambiguity_loss: ambiguity_loss in a min-batch
        """
        ambiguity_loss = 0
        for i, cur_output in enumerate(output):
            diff_avg = (cur_output - pred_avg)
            diff_avg_squ = torch.mul(diff_avg, diff_avg)
            molecule = - torch.sum(diff_avg_squ) / 2
            ambiguity_loss += molecule / len(pred_avg)

        return ambiguity_loss
