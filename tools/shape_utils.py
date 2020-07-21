import torch
import torch_geometric.io
import scipy.io
from scipy import sparse
import numpy as np
from torch_geometric.nn import fps, knn_graph
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from param import *
from arap_potential import arap_vert
import os


def plot_shape(shape):
    """Plot the shape 'shape' which should be scaled to the unit box"""

    vert = shape.get_vert_full_np()

    ax = plt.axes(projection='3d')
    ax.plot_trisurf(vert[:, 0], vert[:, 1], vert[:, 2], triangles=shape.get_triv_np(), cmap='viridis', linewidths=0.2)

    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_zlim(0.2, 0.8)

    plt.show()


def scatter_shape_pair(shape_x, shape_y, velo_x=None, title=None):
    """Plot a pair of shapes as a scatter plot of the vertices"""
	
    vert_x = shape_x.get_vert_np()
    vert_y = shape_y.get_vert_np()

    if not velo_x is None:
        velo_x = velo_x.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vert_x[:, 0], vert_x[:, 1], vert_x[:, 2], marker='o')
    ax.scatter(vert_y[:, 0], vert_y[:, 1], vert_y[:, 2], marker='^')

    if not velo_x is None:
        ax.quiver(vert_x[:, 0], vert_x[:, 1], vert_x[:, 2], velo_x[:, 0], velo_x[:, 1], velo_x[:, 2], length=0.1)

    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_zlim(0.2, 0.8)

    if not title is None:
        plt.title(title)

    plt.show()


def scatter_shape_triplet(shapex, shapey, vert_new, velox=None):
    """Plot a triplet of shapes as a scatter plot of the vertices.
	Those three shapes are X, X^* and Y respectively"""
	
    vertx = shapex.get_vert_np()
    verty = shapey.get_vert_np()
    vert_new = vert_new.detach().cpu().numpy()

    if not velox is None:
        velox = velox.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertx[:, 0], vertx[:, 1], vertx[:, 2], marker='o')
    ax.scatter(verty[:, 0], verty[:, 1], verty[:, 2], marker='^')
    ax.scatter(vert_new[:, 0], vert_new[:, 1], vert_new[:, 2], marker='*')

    if not velox is None:
        ax.quiver(vertx[:, 0], vertx[:, 1], vertx[:, 2], velox[:, 0], velox[:, 1], velox[:, 2], length=0.1)

    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_zlim(0.2, 0.8)

    plt.show()


def plot_sequence(vert_sequence, triv):
    """Plot the whole sequence of an interpolation in individual figures"""

    vert_sequence = vert_sequence.detach().cpu().numpy()
    triv = triv.detach().cpu().numpy()

    for i_vert in range(vert_sequence.shape[0]):
        print("vertices #", i_vert)
        vert = vert_sequence[i_vert, :, :]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vert[:, 0], vert[:, 1], vert[:, 2], triangles=triv, cmap='viridis', linewidths=0.2)

        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)
        ax.set_zlim(0.2, 0.8)

        plt.show()


def load_faust_pair(i, offset=0.5*torch.ones([3], device=device, dtype=torch.float32)):
    """Load a pair of faust shapes. The required format of the file is described below"""

    assert i >= 0 and i < 90, "index out of range for faust"
    
    print(i)

    file_load = data_folder_faust + "FAUST_" + str(i).zfill(3) + "_raw.mat"

    return load_shape_pair(file_load)


def load_shape_pair(file_load, offset=0.5*torch.ones([3], device=device, dtype=torch.float32)):
    """Load a pair of shapes. 'file_load' should contain the path to a .mat file containing
	'vert_x', 'vert_y': The vertices of the input shapes in the format nx3
	'triv_x', 'triv_y': The indices of the triangles in the format mx3 with 
	the matlab indexing convention (indices starting from 1)"""

    mat_dict = scipy.io.loadmat(file_load)

    print("Loaded file ", file_load, "")

    shape_x = Shape(torch.from_numpy(mat_dict["vert_x"].astype(np.float32)).to(device),
                    torch.from_numpy(mat_dict["triv_x"].astype(np.long)).to(device) - 1)
    shape_y = Shape(torch.from_numpy(mat_dict["vert_y"].astype(np.float32)).to(device),
                    torch.from_numpy(mat_dict["triv_y"].astype(np.long)).to(device) - 1)

    if not offset is None:
        shape_x.translate(offset)
        shape_y.translate(offset)

    return shape_x, shape_y


def save_sequence(folder_name, file_name, vert_sequence, shape_x, shape_y, time_elapsed=0):
    """Saves an interpolation sequence to a .mat file"""
	
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    vert_x = shape_x.vert.detach().cpu().numpy()
    vert_y = shape_y.vert.detach().cpu().numpy()
    triv_x = shape_x.triv.detach().cpu().numpy()+1
    triv_y = shape_y.triv.detach().cpu().numpy()+1

    if type(shape_x.samples) is list:
        samples = np.array(shape_x.samples, dtype=np.float32)
    else:
        samples = shape_x.samples.detach().cpu().numpy()

    vert_sequence = vert_sequence.detach().cpu().numpy()

    if shape_x.mahal_cov_mat is None:
        mat_dict = {"vert_x": vert_x, "vert_y": vert_y, "triv_x": triv_x, "triv_y": triv_y,
                    "vert_sequence": vert_sequence, "time_elapsed": time_elapsed, "samples": samples}
    else:
        shape_x.mahal_cov_mat = shape_x.mahal_cov_mat.detach().cpu().numpy()
        mat_dict = {"vert_x": vert_x, "vert_y": vert_y, "triv_x": triv_x, "triv_y": triv_y,
                    "vert_sequence": vert_sequence, "time_elapsed": time_elapsed, "samples": samples,
                    "mahal_cov_mat": shape_x.mahal_cov_mat}

    scipy.io.savemat(folder_name + file_name, mat_dict)


def plot_interpolation(vert_sequence, shape_x, shape_y):
    """Plot the overlap and the intermediate shapes for an interpolation"""

    shape_x.vert = vert_sequence[vert_sequence.shape[0] - 1, :, :]
    scatter_shape_pair(shape_x, shape_y)
    plot_sequence(vert_sequence, shape_x.triv)


class Shape:
    """Class for shapes. (Optional) attributes are:
	vert: Vertices in the format nx3
	triv: Triangles in the format mx3
	samples: Index list of active vertices
	neigh: List of 2-Tuples encoding the adjacency of vertices
	neigh_hessian: Hessian/Graph Laplacian of the shape based on 'neigh'
	mahal_cov_mat: The covariance matrix of our anisotropic arap energy"""

    def __init__(self, vert=None, triv=None):
        self.vert = vert
        self.triv = triv
        self.samples = list(range(vert.shape[0]))
        self.neigh = None
        self.neigh_hessian = None
        self.mahal_cov_mat = None
        
        if not self.triv is None:
            self.triv = self.triv.to(dtype=torch.long)

    def subsample_fps(self, goal_vert):
        assert goal_vert <= self.vert.shape[0], "you cannot subsample to more vertices than n"

        ratio = goal_vert / self.vert.shape[0]
        self.samples = fps(self.vert.detach().to(device_cpu), ratio=ratio).to(device)
        self._neigh_knn()

    def reset_sampling(self):
        self.gt_sampling(self.vert.shape[0])

    def gt_sampling(self, n):
        self.samples = list(range(n))
        self.neigh = None

    def scale(self, factor, shift=True):
        self.vert = self.vert * factor

        if shift:
            self.vert = self.vert + (1-factor)/2

    def get_bounding_box(self):
        max_x, _ = self.vert.max(dim=0)
        min_x, _ = self.vert.min(dim=0)

        return min_x, max_x

    def to_box(self, shape_y):

        min_x, max_x = self.get_bounding_box()
        min_y, max_y = shape_y.get_bounding_box()

        extent_x = max_x-min_x
        extent_y = max_y-min_y

        self.translate(-min_x)
        shape_y.translate(-min_y)

        scale_fac = torch.max(torch.cat((extent_x, extent_y), 0))
        scale_fac = 1./scale_fac

        self.scale(scale_fac, shift=False)
        shape_y.scale(scale_fac, shift=False)

        extent_x = scale_fac*extent_x
        extent_y = scale_fac*extent_y

        self.translate(0.5 * (1 - extent_x))
        shape_y.translate(0.5 * (1 - extent_y))

    def translate(self, offset):
        self.vert = self.vert + offset.unsqueeze(0)

    def get_vert(self):
        return self.vert[self.samples, :]

    def get_vert_shape(self):
        return self.get_vert().shape

    def get_triv(self):
        return self.triv

    def get_triv_np(self):
        return self.triv.detach().cpu().numpy()

    def get_vert_np(self):
        return self.vert[self.samples, :].detach().cpu().numpy()

    def get_vert_full_np(self):
        return self.vert.detach().cpu().numpy()

    def get_neigh(self, num_knn=5):
        if self.neigh is None:
            self.compute_neigh(num_knn=num_knn)

        return self.neigh

    def compute_neigh(self, num_knn=5):
        if len(self.samples) == self.vert.shape[0]:
            self._triv_neigh()
        else:
            self._neigh_knn(num_knn=num_knn)

    def _triv_neigh(self):
        print("Compute triv neigh....")

        self.neigh = torch.cat((self.triv[:, [0, 1]], self.triv[:, [0, 2]], self.triv[:, [1, 2]]), 0)

    def _neigh_knn(self, num_knn=5):
        vert = self.get_vert().detach()
        print("Compute knn....")
        self.neigh = knn_graph(vert.to(device_cpu), num_knn, loop=False).transpose(0, 1).to(device)

    def get_neigh_hessian(self):
        if self.neigh_hessian is None:
            self.compute_neigh_hessian()

        return self.neigh_hessian

    def compute_neigh_hessian(self):

        neigh = self.get_neigh()

        n_vert = self.get_vert().shape[0]

        H = sparse.lil_matrix(1e-3 * sparse.identity(n_vert))

        I = np.array(neigh[:, 0].detach().cpu())
        J = np.array(neigh[:, 1].detach().cpu())
        V = np.ones([neigh.shape[0]])
        U = - V
        H = H + sparse.lil_matrix(sparse.coo_matrix((U, (I, J)), shape=(n_vert, n_vert)))
        H = H + sparse.lil_matrix(sparse.coo_matrix((U, (J, I)), shape=(n_vert, n_vert)))
        H = H + sparse.lil_matrix(sparse.coo_matrix((V, (I, I)), shape=(n_vert, n_vert)))
        H = H + sparse.lil_matrix(sparse.coo_matrix((V, (J, J)), shape=(n_vert, n_vert)))

        self.neigh_hessian = H

    def get_global_rotation(self, vert_to):
        R = arap_vert(vert_to, self.vert, self.get_neigh())
        R = R.mean(dim=0)
        U, _, V = torch.svd(R)
        R = torch.mm(U, V.transpose(0, 1))
        return R

    def rotate(self, R):
        self.vert = torch.mm(self.vert, R.transpose(0, 1))

    def rotate_to(self, shape_y, max_it=100):
        for i in range(max_it):
            self.rotate(self.get_global_rotation(shape_y.vert))

    def to(self, device):
        self.vert = self.vert.to(device)
        self.triv = self.triv.to(device)

    def compute_volume(self):
        return self.compute_volume_shifted(self.vert)

    def compute_volume_shifted(self, vert_t):
        vert_t = vert_t - vert_t.mean(dim=0, keepdim=True)
        vert_triv = vert_t[self.triv, :].to(device_cpu)

        vol_tetrahedra = (vert_triv.det() / 6).to(device)

        return vol_tetrahedra.sum()


if __name__== "__main__":
    print("main of shape_utils.py")
