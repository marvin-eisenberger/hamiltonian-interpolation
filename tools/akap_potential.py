
import torch
from base_tools import *


def akap_energy(vert_t, vert_0, neigh, lambda_reg_len=1e-3):
    velo_vert = vert_t - vert_0
    velo_diff = velo_vert[neigh[:, 0], :] - velo_vert[neigh[:, 1], :]

    vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]
    vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]

    velo_orth = torch.sum(vert_diff_0 * velo_diff, dim=1) + torch.sum(vert_diff_t * velo_diff, dim=1)

    return 1e2 * velo_orth.norm() ** 2 + lambda_reg_len * velo_vert.norm() ** 2


if __name__ == "__main__":
    print("main of akap potential")
