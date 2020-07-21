import torch
import math
import numpy as np
from torch.nn import Parameter
import torch.sparse
from shape_utils import *
from param import device, device_cpu
import quaternion as quat
from base_tools import *


def compute_eigenvectors_3d(vert, k):
    kv = torch.arange(1, k+1, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    vert = vert.unsqueeze(2) * kv * math.pi

    vert_sin = torch.sin(vert)
    vert_cos = torch.cos(vert) * kv

    sin_x = vert_sin[:, 0, :].unsqueeze(2).unsqueeze(3)
    sin_y = vert_sin[:, 1, :].unsqueeze(1).unsqueeze(3)
    sin_z = vert_sin[:, 2, :].unsqueeze(1).unsqueeze(2)

    cos_x = vert_cos[:, 0, :].unsqueeze(2).unsqueeze(3)
    cos_y = vert_cos[:, 1, :].unsqueeze(1).unsqueeze(3)
    cos_z = vert_cos[:, 2, :].unsqueeze(1).unsqueeze(2)

    phi = torch.cat(((cos_x * sin_y * sin_z).unsqueeze(1),
                     (sin_x * cos_y * sin_z).unsqueeze(1),
                     (sin_x * sin_y * cos_z).unsqueeze(1)), 1)

    scale_fac = torch.sqrt(kv.unsqueeze(2) ** 2 + kv.unsqueeze(3) ** 2) ** (-1)

    scale_fac = scale_fac.transpose(1, 3).unsqueeze(4)

    scale_fac = torch.cat((scale_fac.unsqueeze(1).repeat_interleave(k, 1),
                           scale_fac.unsqueeze(2).repeat_interleave(k, 2),
                           scale_fac.unsqueeze(3).repeat_interleave(k, 3)), 5)

    phi = phi.transpose(1, 4).unsqueeze(5).unsqueeze(6)

    phi = torch.sum(hat_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) * phi, 4)
    phi = phi * scale_fac

    phi = phi.transpose(1, 4).reshape(vert.shape[0], 3, -1).transpose(1, 2)

    return phi


def tensor_prod_velocity(phi, a):
    return torch.bmm(phi.permute((2, 0, 1)), a.unsqueeze(0).unsqueeze(2).repeat(3, 1, 1)).permute((1, 2, 0)).squeeze()


def div_free_trans(velo_t, vert_t, k):
    n_feat = 3 * k ** 3

    phi = compute_eigenvectors_3d(vert_t, k)

    M = my_eye(n_feat) * 1e-3

    for d in range(3):
        M = M + torch.mm(phi[..., d].transpose(0, 1), phi[..., d])

    M = M.unsqueeze(0)

    phi = phi.permute([2, 0, 1])

    xi_d = torch.bmm(phi.transpose(1, 2), velo_t.unsqueeze(2).permute([1, 0, 2]))
    xi_d, _ = torch.solve(xi_d, M)
    velo_t = torch.bmm(phi, xi_d)
    velo_t = velo_t.permute([1, 2, 0]).squeeze()

    return velo_t, xi_d


def apply_field(vert_t, xi_d, k):
    phi = compute_eigenvectors_3d(vert_t, k)
    phi = phi.permute([2, 0, 1])

    velo_t = torch.bmm(phi, xi_d)
    velo_t = velo_t.permute([1, 2, 0]).squeeze()

    return velo_t


class Rigid(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.translation = Parameter(torch.zeros([3], dtype=torch.float32, device=device), requires_grad=True)
        self.rotation = Parameter(torch.as_tensor([1, 0, 0, 0], dtype=torch.float32, device=device), requires_grad=True)

    def forward(self, vert):
        vert = quat.qrot((self.rotation / (self.rotation.norm())).repeat(vert.shape[0], 1), vert - 0.5)
        vert = vert + self.translation.unsqueeze(0) + 0.5
        return vert

    def detach_(self):
        self.translation.requires_grad_(False)
        self.rotation.requires_grad_(False)

    def detach(self):
        self.detach_()
        return self


if __name__ == "__main__":
    print("main of div_free_base.py")
