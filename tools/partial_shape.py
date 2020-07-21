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
import os
from shape_utils import *


def load_part_shape_pair(file_load, offset=0.5*torch.ones([3], device=device, dtype=torch.float32)):
    mat_dict = scipy.io.loadmat(file_load)
    """Load a pair of shapes. 'file_load' should contain the path to a .mat file containing
	'vert_x', 'vert_y': The vertices of the input shapes in the format nx3
	'triv_x', 'triv_y': The indices of the triangles in the format mx3 with 
	the matlab indexing convention (indices starting from 1)
	'ass_x', 'ass_y': Two lists of indices (1d arrays) such that 
	ass_x[i] is the point corresponding to ass_y[i] for all i"""
	

    print("Loaded file ", file_load, "")

    shape_x = PartialShape(torch.from_numpy(mat_dict["vert_x"].astype(np.float32)).to(device),
                    torch.from_numpy(mat_dict["triv_x"].astype(np.long)).to(device) - 1,
                    torch.from_numpy(mat_dict["ass_x"].astype(np.long)).to(device) - 1)
    shape_y = PartialShape(torch.from_numpy(mat_dict["vert_y"].astype(np.float32)).to(device),
                    torch.from_numpy(mat_dict["triv_y"].astype(np.long)).to(device) - 1,
                    torch.from_numpy(mat_dict["ass_y"].astype(np.long)).to(device) - 1)

    if not offset is None:
        shape_x.translate(offset)
        shape_y.translate(offset)

    return shape_x, shape_y


class PartialShape(Shape):
    """Class for partial shapes or shapes without equivalent meshing.
	Attributes beyond the base class 'Shape' are:
	ass_full: For a pair of partial shapes shape_x and shape_y,
	shape_x.vert[shape_x.ass[i], :] is the vertex corresponding to 
	shape_y.vert[shape_y.ass[i], :] for all i
	ass: like ass_full, but also adapts to fps subsampling of the shape
	"""

    def __init__(self, vert=None, triv=None, ass=None):
        super().__init__(vert, triv)
        self.ass_full = ass
        self.ass = ass

    def subsample_pairwise(self, goal_vert, shape_y):
        self.subsample_fps(goal_vert)
        _, idx_1, idx_2 = np.intersect1d(self.ass_full.detach().cpu().numpy(), self.samples.detach().cpu().numpy(),
                                     return_indices=True)

        shape_y.ass = shape_y.ass_full[torch.as_tensor(idx_1, device=device)].squeeze()
        self.ass = torch.as_tensor(idx_2, device=device)

    def to_box(self, shape_y):

        min_x, max_x = self.get_bounding_box()
        min_y, max_y = shape_y.get_bounding_box()

        min_xy = torch.min(min_x, min_y)
        max_xy = torch.max(max_x, max_y)

        extent = max_xy-min_xy

        self.translate(-min_xy)
        shape_y.translate(-min_xy)

        scale_fac = torch.max(extent)
        scale_fac = 1./scale_fac

        self.scale(scale_fac, shift=False)
        shape_y.scale(scale_fac, shift=False)

        extent = scale_fac*extent

        self.translate(0.5 * (1 - extent))
        shape_y.translate(0.5 * (1 - extent))


if __name__== "__main__":
    print("main of partial_shape.py")
