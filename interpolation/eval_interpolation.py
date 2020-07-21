from div_free_interpolation import *
from discrete_shell_potential import *
import datetime
import numpy as np
import os
from shape_utils import *
from base_tools import *
from param import *
import matplotlib.pyplot as plt
import scipy.io


def np_to_torch(m, long=False):
    if long:
        return torch.as_tensor(m, dtype=torch.long, device=device)
    else:
        return torch.as_tensor(m, dtype=torch.float32, device=device)


def get_file_array(dataset):
    if dataset == "FAUST":
        file_array = list(range(90))
    elif dataset == "FAUST_sub":
        file_array = list(range(90))


    return file_array


def save_curve(m_array, file_name):
    folder_name = data_folder_out + "curves/"

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    dict_out = {}

    for i_m in range(len(m_array)):
        dict_out["curve_" + str(i_m)] = m_array[i_m]

    scipy.io.savemat(folder_name + file_name, dict_out)


def load_seq_file(folder_name, i_file):
    file_name = "seq_" + str(i_file).zfill(3) + ".mat"

    mat_dict = scipy.io.loadmat(folder_name + file_name)

    vert_sequence = np_to_torch(mat_dict["vert_sequence"])

    if "time_elapsed" in mat_dict.values():
        time_elapsed = mat_dict["time_elapsed"]
    else:
        time_elapsed = -1

    shape_x = Shape(np_to_torch(mat_dict["vert_x"]), np_to_torch(mat_dict["triv_x"], long=True)-1)
    shape_y = Shape(np_to_torch(mat_dict["vert_y"]), np_to_torch(mat_dict["triv_y"], long=True)-1)

    return shape_x, shape_y, vert_sequence, time_elapsed


def plot_curves(m_array, title=None, logarithmic=False):
    num_plot = 200
    coords_array = []

    for m in m_array:
        m_stacked = m.view(-1).cpu()
        m_sort, _ = torch.sort(m_stacked)

        if logarithmic:
            plot_select = np.linspace(0, np.log(m_sort.shape[0] - 1), num_plot, dtype=np.float64)
            plot_select = np.exp(plot_select)
            plot_select = torch.as_tensor(plot_select, device=device_cpu, dtype=torch.long)
            m_sort = m_sort[plot_select]

            y = np.linspace(0, 1, m_sort.shape[0], dtype=np.float64)
            m_sort = m_sort.detach().cpu().numpy()
            plt.semilogx(m_sort, y)
        else:
            plot_select = np.linspace(0, m_sort.shape[0] - 1, num_plot, dtype=np.long)
            plot_select = torch.as_tensor(plot_select, device=device_cpu)
            m_sort = m_sort[plot_select]

            y = np.linspace(0, 1, m_sort.shape[0], dtype=np.float64)
            m_sort = m_sort.detach().cpu().numpy()
            plt.plot(m_sort, y)

        coords_array_curr = np.zeros([y.shape[0], 2], dtype=np.float64)
        coords_array_curr[:, 0] = m_sort
        coords_array_curr[:, 1] = y

        coords_array.append(coords_array_curr)

    if not title is None:
        plt.title(title)

    plt.ylim(0, 1)
    plt.grid()

    plt.show()

    return coords_array


def eval_volume_change(method, dataset, folder_idx, plot=False):
    folder_name = data_folder_out + method + "/" + dataset + "_" + str(folder_idx) + "/"

    file_array = get_file_array(dataset)
    num_files = len(file_array)

    volume_diff = []

    for i_file in file_array:
        shape_x, shape_y, vert_sequence, time_elapsed = load_seq_file(folder_name, i_file)

        num_t = vert_sequence.shape[0]

        volume_diff_curr = my_zeros([num_t]).to(dtype=torch.float64)
        volume_ref = shape_x.compute_volume().to(dtype=torch.float64)

        shape_x_new = deepcopy(shape_x)
        shape_x_new.vert = vert_sequence[num_t-1, ...]

        for t in range(num_t):
            volume_curr = shape_x.compute_volume_shifted(vert_sequence[t, ...]).to(dtype=torch.float64)
            volume_diff_curr[t] = volume_curr / volume_ref + volume_ref / volume_curr - 2

        volume_diff.append(volume_diff_curr)

        if plot:
            scatter_shape_pair(shape_x_new, shape_y, title="Mean volume change: " + str(volume_diff_curr.mean()))

    volume_diff_tens = my_zeros([num_files, num_t]).to(dtype=torch.float64)

    for i_file in range(len(volume_diff)):
        volume_diff_tens[i_file, :] = volume_diff[i_file]

    print("Mean volume change: ", volume_diff_tens.mean())

    return volume_diff_tens


def compute_chamfer(shape_x, shape_y, vert_sequence, num_eval=10000):
    num_t = vert_sequence.shape[0]

    shape_x_new = deepcopy(shape_x)
    shape_x_new.vert = vert_sequence[num_t - 1, ...]

    samples = knn(shape_y.vert.to(device_cpu), shape_x_new.vert.to(device_cpu), k=1).to(device)
    chamfer_curr = (shape_x_new.vert[samples[0, :], :] - shape_y.vert[samples[1, :], :]).norm(dim=1)

    idx_eval = torch.zeros([10000], device=device, dtype=torch.long).random_(0, chamfer_curr.shape[0])
    chamfer_curr = chamfer_curr[idx_eval]

    return chamfer_curr


def eval_chamfer(method, dataset, folder_idx, plot=False):
    folder_name = data_folder_out + method + "/" + dataset + "_" + str(folder_idx) + "/"

    file_array = get_file_array(dataset)
    num_files = len(file_array)
    num_eval = 10000

    chamfer_array = []

    for i_file in file_array:
        shape_x, shape_y, vert_sequence, time_elapsed = load_seq_file(folder_name, i_file)

        chamfer_curr = compute_chamfer(shape_x, shape_y, vert_sequence)

        chamfer_array.append(chamfer_curr)

        if plot:
            scatter_shape_pair(shape_x_new, shape_y, title="Mean chamfer dist: " + str(chamfer_curr.mean()))

    chamfer_tens = my_zeros([num_files, num_eval])

    for i_file in range(len(chamfer_array)):
        chamfer_tens[i_file, :] = chamfer_array[i_file]

    print("Mean chamfer dist: ", chamfer_tens.mean())

    return chamfer_tens


def compute_distortion(shape_x, shape_y, vert_sequence, num_eval=10000):
    dist_max = 10

    num_t = vert_sequence.shape[0]
    num_triv = shape_x.triv.shape[0]

    dist_curr = my_zeros([num_t, num_eval])

    shape_x_new = deepcopy(shape_x)
    shape_x_new.vert = vert_sequence[num_t - 1, ...]

    for t in range(num_t):
        normal_0, _, area_0, _, _, edge_t, edge_proj_0 = discrete_shell_energy_pre(vert_sequence[t, ...], shape_x.vert,
                                                                                   shape_x.triv)
        _, a_membrane_n = membrane_transformation(edge_t, area_0, normal_0, edge_proj_0)
        distortion_curr = (batch_trace(torch.bmm(a_membrane_n.transpose(1, 2), a_membrane_n)).squeeze()) / \
                          (torch.det(a_membrane_n) + 1e-6) - 3
        distortion_curr = torch.abs(distortion_curr)
        distortion_curr = dist_max - torch.relu(dist_max - distortion_curr)

        idx_eval = torch.zeros([10000], device=device, dtype=torch.long).random_(0, num_triv)
        distortion_curr = distortion_curr[idx_eval]

        dist_curr[t, :] = distortion_curr

    return dist_curr


def eval_distortion(method, dataset, folder_idx, plot=False):
    folder_name = data_folder_out + method + "/" + dataset + "_" + str(folder_idx) + "/"

    file_array = get_file_array(dataset)
    num_files = len(file_array)
    num_eval = 10000

    distortion_array = []

    for i_file in file_array:
        shape_x, shape_y, vert_sequence, time_elapsed = load_seq_file(folder_name, i_file)

        num_t = vert_sequence.shape[0]

        dist_curr = compute_distortion(shape_x, shape_y, vert_sequence)

        distortion_array.append(dist_curr)

        if plot:
            scatter_shape_pair(shape_x_new, shape_y, title="Mean distortion: " + str(dist_curr.mean()))

    distortion_tens = my_zeros([num_files, num_t, num_eval])

    for i_file in range(len(distortion_array)):
        distortion_tens[i_file, ...] = distortion_array[i_file]

    print("Mean distortion: ", distortion_tens.mean())

    return distortion_tens


def get_folder_idx(method, dataset):
    if dataset == "FAUST":
        folder_idx = 1
    elif dataset == "FAUST_sub":
        folder_idx = 1

    return folder_idx


def eval_all(dataset, save_results=False):
    print("Evaluate ", dataset, "...")

    distortion_array = []
    volume_array = []
    chamfer_dist_array = []

    for method in ["ham", "div"]:
        print("Method: ", method, "...")

        folder_idx = get_folder_idx(method, dataset)
        folder_idx = str(folder_idx).zfill(3)

        try:
            distortion = eval_distortion(method, dataset, folder_idx)
            volume_change = eval_volume_change(method, dataset, folder_idx)
            chamfer_dist = eval_chamfer(method, dataset, folder_idx)

            distortion_array.append(distortion)
            volume_array.append(volume_change)
            chamfer_dist_array.append(chamfer_dist)
        except Exception as e:
            print("Skipping method ", method, "...")
            print(type(e))
            print(e.args)
            print(e)


    coords_conf_dist = plot_curves(distortion_array, 'Conformal distortion')
    coords_volume = plot_curves(volume_array, 'Volume change', logarithmic=True)
    coords_chamfer = plot_curves(chamfer_dist_array, 'Chamfer distance')

    if save_results:
        save_curve(coords_conf_dist, dataset + "_conf_dist.mat")
        save_curve(coords_volume, dataset + "_volume_change.mat")
        save_curve(coords_chamfer, dataset + "_chamfer_dist.mat")
    return 0


def eval_single(dataset, method):
    folder_idx = get_folder_idx(method, dataset)
    folder_idx = str(folder_idx).zfill(3)

    distortion = eval_distortion(method, dataset, folder_idx)
    volume_change = eval_volume_change(method, dataset, folder_idx)
    chamfer_dist = eval_chamfer(method, dataset, folder_idx)

    plot_curves([distortion], 'Conformal distortion')
    plot_curves([volume_change], 'Volume change', logarithmic=True)
    plot_curves([chamfer_dist], 'Chamfer distance')


if __name__ == "__main__":
    #choose dataset to evaluate
    dataset = "FAUST"
    # dataset = "FAUST_sub"
    
    #choose method to evaluate
    method = "ham"
    eval_single(dataset, method)

