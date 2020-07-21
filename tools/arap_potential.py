
import torch.nn.functional
import torch
from base_tools import *


def arap(vert_diff_t, vert_diff_0, neigh, n_vert):
    norm_t = vert_diff_t.norm(dim=1, keepdim=True)+1e-6
    norm_0 = vert_diff_0.norm(dim=1, keepdim=True)+1e-6

    weight = norm_0 * norm_t

    vert_diff_0 = vert_diff_0 / norm_0
    vert_diff_t = vert_diff_t / norm_t

    cross = torch.bmm(hat_op(vert_diff_0), vert_diff_t.unsqueeze(2)).squeeze()

    # compute per-point quantities
    cross_pp = my_zeros([n_vert, 3])
    weight_pp = my_zeros([n_vert, 1])

    weight_pp = torch.index_add(weight_pp, 0, neigh[:, 0], weight)
    weight_pp = torch.index_add(weight_pp, 0, neigh[:, 1], weight)

    cross = cross * weight
    cross_pp = torch.index_add(cross_pp, 0, neigh[:, 0], cross)
    cross_pp = torch.index_add(cross_pp, 0, neigh[:, 1], cross)
    cross_pp = cross_pp / weight_pp

    # compute rotation matrix
    hat_cross_pp = hat_op(cross_pp)

    sin_alpha = hat_cross_pp.norm(dim=1, keepdim=True)
    cos_alpha = torch.sqrt(torch.abs(1-sin_alpha**2))

    R = my_eye(3).unsqueeze(0) + hat_cross_pp + torch.bmm(hat_cross_pp, hat_cross_pp) * 1 / (1 + cos_alpha)

    return R


def arap_vert(vert_t, vert_0, neigh):
    n_vert = vert_t.shape[0]

    vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
    vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]

    return arap(vert_diff_t, vert_diff_0, neigh, n_vert)


def arap_energy(vert_t, vert_0, neigh, lambda_reg_len=1e-6):
    n_vert = vert_t.shape[0]

    vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
    vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]

    R_t = arap(vert_diff_t, vert_diff_0, neigh, n_vert)

    R_neigh_t = 0.5 * (torch.index_select(R_t, 0, neigh[:, 0]) + torch.index_select(R_t, 0, neigh[:, 1]))

    vert_diff_0_rot = torch.bmm(R_neigh_t, vert_diff_0.unsqueeze(2)).squeeze()
    acc_t_neigh = vert_diff_t - vert_diff_0_rot

    E_arap = acc_t_neigh.norm()**2 + lambda_reg_len*(vert_t-vert_0).norm()**2

    return E_arap


if __name__ == "__main__":
    print("main of arap potential")
