
import torch
from base_tools import *


def w_membrane(a_membrane, a_membrane_n, param):
    lambd = param.lambd
    mu = param.mu

    a_membrane_n_det = torch.det(a_membrane_n).unsqueeze(1).unsqueeze(2)
    a_membrane_n_det = soft_relu(a_membrane_n_det, 1e-4)

    w = mu/2 * batch_trace(a_membrane) + lambd/4 * a_membrane_n_det - \
        (2*mu+lambd)/4 * torch.log(a_membrane_n_det) - mu - lambd/4

    return w


def membrane_transformation(edge_t, area_0, normal_0, edge_proj_0):
    edge_norm_t = torch.norm(edge_t, dim=2, keepdim=True)  # [n, 3, 1]: #tri x #edge x 1
    edge_norm_proj_t = torch.matmul(edge_norm_to_proj.unsqueeze(0), edge_norm_t ** 2).unsqueeze(
        3)  # [n, 3, 1, 1]: #tri x #edge x 1 x 1

    a_membrane = 1 / soft_relu(2 * area_0 ** 2) * torch.sum(edge_norm_proj_t * edge_proj_0,
                                                           dim=1)  # [n, 3, 3]: #tri x (#dim1 x #dim2)

    a_membrane_n = a_membrane + (
            normal_0 * normal_0.transpose(1, 2))  # [n, 3, 3]: #tri x (#dim1 x #dim2)

    return a_membrane, a_membrane_n


def w_fff(normal_0, area_0, edge_t, edge_proj_0, param):

    a_membrane, a_membrane_n = membrane_transformation(edge_t, area_0, normal_0, edge_proj_0)

    w = w_membrane(a_membrane, a_membrane_n, param)  # [n, 1, 1]: #tri x 1 x 1

    return w


def vertex_normals(face_normal, triv, n_vert):
    vert_normal = my_zeros([n_vert, 3])
    for d in range(3):
        vert_normal = vert_normal.index_add(0, triv[:, d], face_normal.squeeze())

    vert_normal = vert_normal / (vert_normal.norm(dim=1, keepdim=True)+1e-5)

    return vert_normal


def q_sff(normal, area_0, edge, edge_proj_0, triv, n_vert):
    vert_normal = vertex_normals(normal, triv, n_vert)  # [n_vert, 3]: #vert x #dim
    vert_normal_triv = vert_normal[triv]  # [n, 3, 3]: #tri x #corner x #dim

    vert_normal_diff = torch.matmul(triv_to_edge.unsqueeze(0), vert_normal_triv)  # [n, 3, 3]: #tri x #edge x #dim

    normal_diff_t = torch.sum(vert_normal_diff * edge, dim=2, keepdim=True)  # [n, 3, 1]: #tri x #edge x 1

    normal_proj_t = torch.matmul(edge_norm_to_proj.unsqueeze(0), normal_diff_t).unsqueeze(3)  # [n, 3, 1, 1]: #tri x #edge x 1 x 1

    q = 1/soft_relu(8 * area_0**2) * torch.sum(normal_proj_t * edge_proj_0, dim=1)  # [n, 3, 3]: #tri x (#dim1 x #dim2)
    return q


def discrete_shell_energy_pre(vert_t, vert_0, triv):

    vert_triv_0 = vert_0[triv]  # [n, 3, 3]: #tri x #corner x #dim
    vert_triv_t = vert_t[triv]  # [n, 3, 3]: #tri x #corner x #dim

    edge_0 = 1e2 * torch.matmul(triv_to_edge.unsqueeze(0), vert_triv_0)  # [n, 3, 3]: #tri x #edge x #dim
    edge_t = 1e2 * torch.matmul(triv_to_edge.unsqueeze(0), vert_triv_t)  # [n, 3, 3]: #tri x #edge x #dim

    normal_0 = cross_prod(edge_0[:, 0, :], edge_0[:, 1, :])  # [n, 3, 1]: #tri x #dim x 1
    normal_t = cross_prod(edge_t[:, 0, :], edge_t[:, 1, :])  # [n, 3, 1]: #tri x #dim x 1
    area_0 = soft_relu(normal_0.norm(dim=1, keepdim=True))  # [n, 1, 1]: #tri x 1 x 1
    area_t = soft_relu(normal_t.norm(dim=1, keepdim=True))  # [n, 1, 1]: #tri x 1 x 1

    normal_0 = normal_0 / area_0  # [n, 3, 1]: #tri x #dim x 1
    normal_t = normal_t / area_t  # [n, 3, 1]: #tri x #dim x 1

    edge_proj_0 = cross_prod(normal_0.squeeze(), edge_0.transpose(1, 2)).transpose(1,
                                                                                   2)  # [n, 3, 3]: #tri x #edge x #dim
    edge_proj_0 = edge_proj_0.unsqueeze(2) * edge_proj_0.unsqueeze(3)  # [n, 3, 3, 3]: #tri x #edge x (#dim1 x #dim2)

    return normal_0, normal_t, area_0, area_t, edge_0, edge_t, edge_proj_0


def discrete_shell_energy(vert_t, vert_0, triv, param):

    n_vert = vert_t.shape[0]

    normal_0, normal_t, area_0, area_t, edge_0, edge_t, edge_proj_0 = discrete_shell_energy_pre(vert_t, vert_0, triv)

    # ----------------------------------------------------

    w = w_fff(normal_0, area_0, edge_t, edge_proj_0, param)  # [n, 1, 1]: #tri x 1 x 1
    E_membrane = torch.sum(1e-4*(w * area_0), dim=0).squeeze()

    # ----------------------------------------------------

    q_0 = q_sff(normal_0, area_0, edge_0, edge_proj_0, triv, n_vert)
    q_t = q_sff(normal_t, area_0, edge_t, edge_proj_0, triv, n_vert)

    q = torch.sum((q_t - q_0)**2, dim=(1, 2), keepdim=True)  # [n, 1, 1]: #tri x 1 x 1
    E_bending = torch.sum(1e-4 * (q * area_0), dim=0).squeeze()

    E_total = E_membrane + param.nu * E_bending

    return E_total


if __name__ == "__main__":
    print("main of discrete_shell_potential.py")
