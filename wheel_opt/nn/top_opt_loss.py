import torch
import torch.nn as nn


class TopOptLoss(nn.Module):

    def __init__(self):
        super(TopOptLoss, self).__init__()

    def forward(self, density_new_tt, objective, volume_fraction, penal, psi_0, alpha, volumes):
        objective = torch.sum(torch.div(objective, density_new_tt**penal)) / psi_0
        vol_constraint = torch.sum(density_new_tt * volumes) / volume_fraction - 1.0

        return objective + alpha * torch.pow(vol_constraint, 2), vol_constraint
