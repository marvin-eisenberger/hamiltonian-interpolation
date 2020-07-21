import torch
import math
import time
import numpy as np
from torch.nn import Linear, Parameter
import torch.nn.functional
from shape_utils import *
from param import device, device_cpu
import quaternion as quat
from torch_geometric.nn import knn
import matplotlib.pyplot as plt
from div_free_base import *
from datetime import datetime
from arap_potential import *
from interpolation_base import *
from multiscale_shape import *
from partial_shape import *


class DivFreeParam(Param):
    """Set of hyperparameters for the div-free method"""

    def __init__(self):
        super().__init__()
        self.lr = 0.02
        self.scales = [12500]

        self.num_vert = 2000
        self.num_knn = 5
        self.k = 8
        self.num_it = 100
        self.num_timesteps = 16
        self.step_size = 1 / self.num_timesteps
        self.lambda_reg = 20
        self.lambda_inertia = 0.3
        self.extrapolate_velo = True
        self.rigid = False

        self.log = True


class DivFreeModuleBase(InterpolationModuleMultiscaleBase):
    """Base class of methods based on a low rank div-free vector field representation"""

    def __init__(self, shape_x, shape_y, param, a=None):
        super().__init__()

        self.param = param
        self.shape_x = shape_x
        self.shape_y = shape_y
        
        if a is None:
            self.a = Parameter(torch.zeros([3 * param.k ** 3], dtype=torch.float32, device=device), requires_grad=True)
        else:
            self.a = Parameter(a, requires_grad=True)
        
        if param.rigid:
            self.rigid = Rigid()
        else:
            self.rigid = None

        self.assignment_x = None
        self.assignment_y = None

        self.E_align_log = []
        self.E_reg_log = []

        self.update_correspondence_gt()

    def forward(self):

        vert_new, E_reg = self._forward_integration()

        E_reg = self.param.lambda_reg * E_reg

        E_align = self.get_alignment_error(self.shape_y, vert_new)
        E = E_align + E_reg

        self.E_align_log.append(E_align.detach())
        self.E_reg_log.append(E_reg.detach())

        return E, [E_align, E_reg]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        return self

    def get_vert_sequence(self):
        hist = self._get_hist()

        rigid = self.rigid
        a = self.a.detach()
        if not rigid is None:
            rigid = rigid.detach()

        self._forward_integration(hist=hist)

        num_vert = self.shape_x.vert.shape[0]
        samples = list(range(num_vert))

        sub_indices = list(range(0, num_vert, 1000))
        sub_indices.append(num_vert)

        vert_sequence = my_zeros([self.param.num_timesteps+1, self.shape_x.vert.shape[0], 3])

        samples_x_copy = self.shape_x.samples

        for i in range(len(sub_indices) - 1):
            indices = slice(sub_indices[i], sub_indices[i + 1])
            self.shape_x.samples = samples[indices]

            vert_sequence[:, indices, :] = self._apply_vector_field(self.shape_x.vert[indices, :], hist["xi_hist_out"], a, self.param, rigid)

        self.shape_x.samples = samples_x_copy
        return vert_sequence
        
    def _get_hist(self):
        return {"vert_hist_out": [], "velo_hist_out": [], "xi_hist_out": [], "vert_mean_hist_out": []}

    def _apply_vector_field(self, vert_0, xi_hist, a, param, rigid=None):
        raise NotImplementedError()

    def _forward_integration(self, hist=None):
        raise NotImplementedError()

    def get_alignment_error(self, shape_y, vert_new):
        return torch.nn.functional.mse_loss(vert_new[self.assignment_x, :], shape_y.get_vert()[self.assignment_y, :])

    def update_correspondence_gt(self):
        self.assignment_x = slice(self.shape_x.get_vert().shape[0])
        self.assignment_y = slice(self.shape_x.get_vert().shape[0])

    def update_correspondence_descriptor(self):
        raise NotImplementedError()

    def step_multiscale(self, i_scale):
        self.shape_x.increase_scale_idx()
        self.shape_y.increase_scale_idx()

        return self.copy_self(self.a.data)

    def copy_self(self, a=None):
        return DivFreeModuleBase(self.shape_x, self.shape_y, self.param, a)


class HamiltonianModule(DivFreeModuleBase):
    """Module for Hamiltonian shape interpolation"""
	
    def __init__(self, shape_x, shape_y, param, a=None):
        super().__init__(shape_x, shape_y, param, a)

        self.acc = Parameter(torch.zeros([shape_x.get_vert_shape()[0], 3, 3], dtype=torch.float32, device=device) + my_eye(3).unsqueeze(0), requires_grad=True)
        self.log_lambda_inertia = torch.as_tensor(math.log(param.lambda_inertia), dtype=torch.float32, device=device)
        self.log_lambda_inertia = Parameter(self.log_lambda_inertia, requires_grad=True)

        self.compute_hessian()

    #override
    def _forward_integration(self, hist=None):
        shape_x = self.shape_x
        a = self.a
        acc = self.acc
        param = self.param
        rigid = self.rigid
        hessian = self.hessian
        log_lambda_inertia = self.log_lambda_inertia
        k = param.k
        steps = param.num_timesteps
        step_size = self.param.step_size

        vert_t = shape_x.get_vert()
        neigh = shape_x.get_neigh(self.param.num_knn)

        if not rigid is None:
            vert_t = rigid(vert_t)
        vert_0 = vert_t.clone()

        phi = compute_eigenvectors_3d(vert_t, k)
        velo_t = step_size * tensor_prod_velocity(phi, a)
        velo_tm1 = velo_t

        Earap = 0

        for it in range(steps):
            if param.log:
                print('sub - it: ', it, end='\r')

            vert_t, velo_t, velo_tm1, acc_t, xi_t, vert_mean_t = self._compute_step(vert_t, velo_t, velo_tm1, log_lambda_inertia, acc, vert_0, neigh, hessian, param)

            scaling_fac = vert_t.shape[0] * steps * 25 / param.num_knn
            Earap = Earap + torch.sum(acc_t ** 2) / scaling_fac

            if not hist is None:
                hist["vert_hist_out"].append(vert_t.detach().clone().detach())
                hist["velo_hist_out"].append(velo_t.detach().clone().detach())
                hist["xi_hist_out"].append(xi_t.detach().clone().detach())
                hist["vert_mean_hist_out"].append(vert_mean_t.detach().clone().detach())

        return vert_t, Earap

    #override
    def _apply_vector_field(self, vert_0, xi_hist, a, param, rigid=None):
        k = param.k
        steps = param.num_timesteps
        step_size = self.param.step_size

        vert_t = vert_0

        if not rigid is None:
            vert_t = rigid(vert_t)

        phi = compute_eigenvectors_3d(vert_t, k)
        velo_t = step_size * tensor_prod_velocity(phi, a)
        velo_tm1 = velo_t

        vert_sequence = my_zeros([steps + 1, vert_0.shape[0], 3])
        vert_sequence[0, ...] = vert_0.clone()

        for it in range(steps):
            xi_t = xi_hist[it].repeat(3, 1, 1)

            if param.extrapolate_velo:
                velo_t_star = 2 * velo_t - velo_tm1
            else:
                velo_t_star = velo_t

            velo_tm1 = velo_t
            velo_t = apply_field(vert_t + step_size / 2 * velo_t_star, xi_t, k)
            vert_t = vert_t + step_size * velo_t
            vert_sequence[it + 1, ...] = vert_t.clone()

        self.shape_x.mahal_cov_mat = self.acc

        return vert_sequence

    def _compute_acceleration(self, vert_t, vert_tp1, vert_0, neigh, acc, step_size):
        n_vert = vert_t.shape[0]

        vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
        vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]

        vert_diff_tp1 = vert_tp1[neigh[:, 0], :] - vert_tp1[neigh[:, 1], :]

        R_tp1 = arap(vert_diff_tp1, vert_diff_0, neigh, n_vert)

        R_neigh_tp1 = 0.5 * (torch.index_select(R_tp1, 0, neigh[:, 0]) + torch.index_select(R_tp1, 0, neigh[:, 1]))

        vert_diff_0_rot = torch.bmm(R_neigh_tp1, vert_diff_0.unsqueeze(2)).squeeze()
        acc_t_neigh = vert_diff_t - vert_diff_0_rot

        R_tp1 = torch.bmm(R_tp1, acc.transpose(1, 2))

        R_neigh_tp1 = 0.5 * (torch.index_select(R_tp1, 0, neigh[:, 0]) + torch.index_select(R_tp1, 0, neigh[:, 1]))
        acc_t_neigh = torch.bmm(R_neigh_tp1.transpose(1, 2), acc_t_neigh.unsqueeze(2)).squeeze()

        acc_t = torch.zeros(vert_t.shape, dtype=torch.float32, device=device)
        acc_t = torch.index_add(acc_t, 0, neigh[:, 0], -acc_t_neigh)
        acc_t = torch.index_add(acc_t, 0, neigh[:, 1], acc_t_neigh)

        acc_t = torch.bmm(R_tp1, acc_t.unsqueeze(2)).squeeze()

        return acc_t, vert_diff_0_rot, R_tp1

    def _compute_velocity(self, vert_t, velo_t, acc_t, R_tp1, hessian, lambda_inertia, param):
        num_feat = 3 * param.k ** 3
        num_vert = vert_t.shape[0]
        steps = param.num_timesteps
        step_size = param.step_size

        vert_mean_t = vert_t.mean(dim=0, keepdim=True) + 0.5

        phi = compute_eigenvectors_3d(vert_t + step_size / 2 * velo_t.transpose(0, 1).squeeze(), param.k)

        c = torch.bmm(phi.permute([2, 1, 0]), lambda_inertia * velo_t + 0.5 * step_size * acc_t)
        c = c.sum(0)

        phi = phi.permute([2, 0, 1])
        phi_end = phi

        M = 1e-3 * my_eye(num_feat) + lambda_inertia * torch.sum(torch.bmm(phi_end.transpose(1, 2), phi_end), dim=0)

        phi = phi.permute([1, 2, 0])
        phi = torch.bmm(R_tp1.transpose(1, 2), phi.transpose(1, 2))
        phi = phi.permute([1, 2, 0])

        phi_2 = torch.mm(hessian, phi.transpose(0, 2).reshape(num_vert, -1)).reshape(num_vert, -1, 3).permute([2, 0, 1])
        M = M + step_size ** 2 * torch.sum(torch.bmm(phi, phi_2), dim=0)

        xi_t, _ = torch.solve(c, M)

        xi_t = xi_t.unsqueeze(0)

        velo_t = (xi_t * phi_end.permute([0, 2, 1])).sum(1).transpose(0, 1)

        return velo_t, xi_t, vert_mean_t

    def _compute_step(self, vert_t, velo_t, velo_tm1, log_lambda_inertia, acc, vert_0, neigh, hessian, param):

        steps = param.num_timesteps
        step_size = param.step_size
        lambda_inertia = torch.exp(log_lambda_inertia) * step_size

        if param.extrapolate_velo:
            velo_t_star = 2 * velo_t - velo_tm1
        else:
            velo_t_star = velo_t
        velo_tm1 = velo_t

        # compute acceleration
        acc_t, vert_diff_0_rot, R_tp1 = self._compute_acceleration(vert_t, vert_t + step_size * velo_t_star, vert_0, neigh,
                                                             acc, step_size)

        # compute linear system for xi_t
        velo_t_star = velo_t_star.permute([1, 0]).unsqueeze(2)
        acc_t = acc_t.transpose(0, 1).unsqueeze(2)

        velo_t, xi_t, vert_mean_t = self._compute_velocity(vert_t, velo_t_star, acc_t, R_tp1, hessian, lambda_inertia, param)

        vert_t = vert_t + step_size * velo_t

        vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
        
        acc_t = vert_diff_t - vert_diff_0_rot

        return vert_t, velo_t, velo_tm1, acc_t, xi_t, vert_mean_t

    def compute_hessian(self):

        neigh = self.shape_x.get_neigh()

        n_neigh = neigh.shape[0]
        n_vert = self.shape_x.get_vert_shape()[0]

        I1 = torch.cat((neigh[:, 0], neigh[:, 1], neigh[:, 0], neigh[:, 1]), 0)
        I2 = torch.cat((neigh[:, 1], neigh[:, 0], neigh[:, 0], neigh[:, 1]), 0)
        V = torch.cat((-my_ones([n_neigh*2]), my_ones([n_neigh*2])), 0)
        I = torch.cat((I1.unsqueeze(0), I2.unsqueeze(0)), 0)

        self.hessian = torch.sparse.FloatTensor(I, V, (n_vert, n_vert))

        self.hessian = self.hessian.to_dense()

    def copy_self(self, a=None):
        return HamiltonianModule(self.shape_x, self.shape_y, self.param, a)


class HamiltonianExtrapolationModule(HamiltonianModule):
    """Module for Hamiltonian shape interpolation"""

    def __init__(self, shape_x, shape_y, param, a=None, time_max=1.5):
        super().__init__(shape_x, shape_y, param, a)

        self.param.extrapolate_velo = False

        self.step_eval = int(param.num_timesteps/time_max)
        self.time_max = time_max

    #override
    def get_vert_sequence(self):
        hist = self._get_hist()

        rigid = self.rigid
        self.a.detach_()
        self.log_lambda_inertia.detach_()
        self.acc.detach_()

        if not rigid is None:
            rigid = rigid.detach()

        self.param.num_timesteps = int(self.param.num_timesteps * self.time_max)

        a = self.a

        print("number of steps: ", self.param.num_timesteps)

        self._forward_integration(hist=hist)

        num_vert = self.shape_x.vert.shape[0]
        samples = list(range(num_vert))

        sub_indices = list(range(0, num_vert, 1000))
        sub_indices.append(num_vert)

        vert_sequence = my_zeros([self.param.num_timesteps+1, self.shape_x.vert.shape[0], 3])

        samples_x_copy = self.shape_x.samples

        for i in range(len(sub_indices) - 1):
            indices = slice(sub_indices[i], sub_indices[i + 1])
            self.shape_x.samples = samples[indices]

            vert_sequence[:, indices, :] = self._apply_vector_field(self.shape_x.vert[indices, :], hist["xi_hist_out"], a, self.param, rigid)

        self.shape_x.samples = samples_x_copy
        return vert_sequence


class HamiltonianPartialModule(HamiltonianModule):
    """Module for Hamiltonian shape interpolation with partial shapes"""

    def __init__(self, shape_x: PartialShape, shape_y: PartialShape, param, a=None):
        super().__init__(shape_x, shape_y, param, a)

    #override
    def update_correspondence_gt(self):
        self.assignment_x = self.shape_x.ass
        self.assignment_y = self.shape_y.ass


class DivFreeStationaryModule(DivFreeModuleBase):
    """Stationary Divergence-Free interpolation"""

    def __init__(self, shape_x, shape_y, param, a=None):
        super().__init__(shape_x, shape_y, param, a)

    #override
    def _forward_integration(self, hist=None):
        shape_x = self.shape_x
        a = self.a
        param = self.param
        rigid = self.rigid
        k = param.k
        steps = param.num_timesteps
        step_size = self.param.step_size

        vert_t = shape_x.get_vert()
        neigh = shape_x.get_neigh(self.param.num_knn)

        if not rigid is None:
            vert_t = rigid(vert_t)
        vert_0 = vert_t.clone()

        xi = a.unsqueeze(0).unsqueeze(2)

        Earap = 0

        for it in range(steps):
            if param.log:
                print('sub - it: ', it, end='\r')

            phi = step_size * compute_eigenvectors_3d(vert_t, k)
            velo_t = tensor_prod_velocity(phi, a)
            vert_mid_t = vert_t + 0.5 * step_size * velo_t

            phi = step_size * compute_eigenvectors_3d(vert_mid_t, k)
            velo_t = tensor_prod_velocity(phi, a)
            vert_t = vert_mid_t + step_size * velo_t

            scaling_fac = vert_t.shape[0] * steps * 5 * 5 / param.num_knn
            Earap = Earap + arap_energy(vert_t, vert_0, neigh, lambda_reg_len=0) / scaling_fac

            if not hist is None:
                hist["vert_hist_out"].append(vert_t.detach().clone().detach())
                hist["velo_hist_out"].append(velo_t.detach().clone().detach())
                hist["xi_hist_out"].append(xi.detach().clone().detach())

        return vert_t, Earap

    #override
    def _apply_vector_field(self, vert_0, xi_hist, a, param, rigid=None):
        k = param.k
        steps = param.num_timesteps
        step_size = self.param.step_size

        vert_t = vert_0

        if not rigid is None:
            vert_t = rigid(vert_t)

        vert_sequence = my_zeros([steps + 1, vert_0.shape[0], 3])
        vert_sequence[0, ...] = vert_0.clone()

        for it in range(steps):
            phi = step_size * compute_eigenvectors_3d(vert_t, k)
            velo_t = tensor_prod_velocity(phi, a)
            vert_mid_t = vert_t + 0.5 * step_size * velo_t

            phi = step_size * compute_eigenvectors_3d(vert_mid_t, k)
            velo_t = tensor_prod_velocity(phi, a)
            vert_t = vert_mid_t + step_size * velo_t

            vert_sequence[it + 1, ...] = vert_t.clone()

        return vert_sequence

    def copy_self(self, a=None):
        return DivFreeStationaryModule(self.shape_x, self.shape_y, self.param, a)



if __name__ == "__main__":
    print("main of div_free_interpolation.py")

