import torch
from shape_utils import Shape, load_shape_pair, scatter_shape_pair
from torch_geometric.nn import knn
from param import *
from arap_potential import arap_vert


def load_multiscale_shapes(folder_path, file_name, scales, offset=0.5*torch.ones([3], device=device, dtype=torch.float32)):
    """Like 'load_shape_pair' but for shapes with different resolutions"""
    vert_x_array = []
    triv_x_array = []

    vert_y_array = []
    triv_y_array = []

    for i_scale in range(len(scales)):
        file_load = folder_path + "sub_" + str(scales[i_scale]) + "/" + file_name
        shape_x, shape_y = load_shape_pair(file_load, offset)

        vert_x_array.append(shape_x.vert)
        vert_y_array.append(shape_y.vert)
        triv_x_array.append(shape_x.triv)
        triv_y_array.append(shape_y.triv)

    shape_x = MultiscaleShape(vert_x_array, triv_x_array)
    shape_y = MultiscaleShape(vert_y_array, triv_y_array)

    return shape_x, shape_y


class MultiscaleShape(Shape):
    """Class for shapes with multiple resolutions.
	Attributes beyond the base class 'Shape' are:
	vert_array: List of vertices with different resolutions
	triv_array: List of triangles with different resolutions
	scale_idx: The index describing the current resolution --
	The current vertices are vert_array[scale_idx]
	ass_[array/vecs/weights]: attributes needed to apply an interpolation
	on scale 'scale_idx' to the next resolution '(scale_idx+1)'
	"""

    def __init__(self, vert_array, triv_array):
        super().__init__(vert_array[0], triv_array[0])

        self.vert_array = vert_array
        self.triv_array = triv_array

        self.scale_idx = 0
        self.scale_idx_len = len(vert_array)

        self.ass_array = None
        self.ass_vecs = None
        self.ass_weights = None
        self.init_upscale()

    def set_scale_idx(self, scale_idx):
        assert scale_idx >= 0 and scale_idx < self.scale_idx_len, "new index out of bounds"

        self.vert_array[self.scale_idx] = self.vert

        self.scale_idx = scale_idx

        self.vert = self.vert_array[scale_idx]
        self.triv = self.triv_array[scale_idx]
        self.samples = list(range(self.vert.shape[0]))
        self.neigh = None

    def increase_scale_idx(self):
        self.set_scale_idx(self.scale_idx+1)

    def next_resolution(self):
        return self.vert_array[self.scale_idx+1].shape

    def init_upscale(self, num_knn=3):
        self.ass_array = []
        self.ass_vecs = []
        self.ass_weights = []
        for idx in range(self.scale_idx_len-1):
            vert_i = self.vert_array[idx].to(device_cpu)
            vert_ip1 = self.vert_array[idx+1].to(device_cpu)

            ass_curr = knn(vert_i, vert_ip1, num_knn)
            ass_curr = ass_curr[1, :].view(-1, num_knn)
            self.ass_array.append(ass_curr.to(device))  #[n_vert_tp1, num_knn]

            vec_curr = vert_ip1.unsqueeze(1) - vert_i[ass_curr, :]
            self.ass_vecs.append(vec_curr.to(device))  #[n_vert_tp1, num_knn, 3]

            weights_curr = 1/(torch.norm(vec_curr, dim=2, keepdim=True)+1e-5)
            weights_curr = weights_curr / torch.sum(weights_curr, dim=1, keepdim=True)
            self.ass_weights.append(weights_curr.to(device))  #[n_vert_tp1, num_knn, 1]

    def apply_upsampling(self, vert_t):
        R = arap_vert(vert_t, self.vert, self.get_neigh())  #[n_vert_tp1, 3, 3]

        ass_curr = self.ass_array[self.scale_idx]
        vec_curr = self.ass_vecs[self.scale_idx]
        weights_curr = self.ass_weights[self.scale_idx]

        vert_tp1 = vert_t[ass_curr, :] + torch.matmul(R[ass_curr], vec_curr.unsqueeze(3)).squeeze()  #[n_vert_tp1, num_knn, 3]
        vert_tp1 = torch.sum(weights_curr * vert_tp1, dim=1)

        return vert_tp1

    def rotate(self, R):
        for i in range(self.scale_idx_len):
            self.vert_array[i] = torch.mm(self.vert_array[i], R.transpose(0, 1))

        self.vert = self.vert_array[self.scale_idx]

        self.init_upscale()

    def to_box(self, shape_y):
        scale_idx = self.scale_idx
        for i in range(self.scale_idx_len):
            self.set_scale_idx(i)
            shape_y.set_scale_idx(i)
            super().to_box(shape_y)

        self.set_scale_idx(scale_idx)
        shape_y.set_scale_idx(scale_idx)

        self.init_upscale()

    def scale(self, factor, shift=True):
        scale_idx = self.scale_idx
        for i in range(self.scale_idx_len):
            self.set_scale_idx(i)
            super().scale(factor, shift)

        self.set_scale_idx(scale_idx)

        self.init_upscale()


if __name__ == "__main__":
    print("main of multiscale_shape.py")

