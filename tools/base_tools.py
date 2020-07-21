
import torch
from param import device
	
	
triv_to_edge = torch.as_tensor([[-1, 1, 0], [0, -1, 1], [1, 0, -1]], dtype=torch.float32, device=device)
edge_norm_to_proj = torch.as_tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=torch.float32, device=device)
hat_matrix = torch.as_tensor([[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                              [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                              [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]], device=device, dtype=torch.float32)


def my_speye(n, offset=1):
    V = my_ones([n])
    I = torch.arange(n)
    I = torch.cat((I.unsqueeze(0), I.unsqueeze(0)), 0).to(dtype=torch.long, device=device)
    V = V * offset
    M = torch.sparse.FloatTensor(I, V, (n, n))
    return M
	

def my_ones(shape):
    return torch.ones(shape, device=device, dtype=torch.float32)

	
def my_zeros(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def my_eye(n):
    return torch.eye(n, device=device, dtype=torch.float32)


def hat_op(v):
    assert v.shape[1] == 3, "wrong input dimensions"

    w = my_zeros([3, 3, 3])

    w[0, 1, 2] = -1
    w[0, 2, 1] = 1
    w[1, 0, 2] = 1
    w[1, 2, 0] = -1
    w[2, 0, 1] = -1
    w[2, 1, 0] = 1

    v = v.transpose(0, 1).unsqueeze(2).unsqueeze(3)
    w = w.unsqueeze(1)

    M = v * w
    M = M.sum(0)

    return M


def cross_prod(u, v):
    if len(v.shape) == 2:
        v = v.unsqueeze(2)
    return torch.bmm(hat_op(u), v)


def batch_trace(m):
    m = (m * my_eye(m.shape[1]).unsqueeze(0)).sum(dim=(1, 2))
    return m.unsqueeze(1).unsqueeze(2)


def soft_relu(m, eps=1e-7):
    return torch.relu(m)+eps

if __name__ == "__main__":
    print("main of base_tools.py")
