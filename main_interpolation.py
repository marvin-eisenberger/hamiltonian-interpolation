
import sys

sys.path.insert(0, './tools')
sys.path.insert(0, './interpolation')

from div_free_interpolation import *
import datetime
import numpy as np
import os


log=True


def get_param(method):
    """Returns the parameter set for a given method"""

    if method == "ham":
        param = DivFreeParam()
    elif method == "div":
        param = DivFreeParam()
    elif method == "extra":
        param = DivFreeParam()
    else:
        raise ValueError()

    param.log = log

    return param


def run_single(shape_x, shape_y, method, rigid_align=True, param=None):
    """Main function to execute a shape interpolation method for two given
	input shapes shape_x and shape_y"""

    if param is None:
        param = get_param(method)

    if rigid_align:
        shape_x.rotate_to(shape_y)
    shape_x.to_box(shape_y)
    shape_x.scale(0.9)
    shape_y.scale(0.9)

    if param.num_vert >= shape_x.vert.shape[0]:
        print("Abort Euclidean fps, number of vertices (" + str(shape_x.vert.shape[0]) +
              ") <= than supposed number of samples (" + str(param.num_vert) + ").")
    else:
        shape_x.subsample_fps(param.num_vert)
        shape_y.samples = shape_x.samples

    if method == "ham":
        interpol_module = HamiltonianModule(shape_x, shape_y, param)
    elif method == "div":
        interpol_module = DivFreeStationaryModule(shape_x, shape_y, param)
    elif method == "extra":
        interpol_module = HamiltonianExtrapolationModule(shape_x, shape_y, param, time_max=10)
    else:
        raise ValueError()
		
    interpol = InterpolationGD(interpol_module, param)
    num_it_super = len(param.scales)

    print(param.__dict__)

    if subsample:
        E = interpol.interpolate()
    else:
        interpol_multi = InterpolationMultiscale(interpol, param, num_it_super=num_it_super)
        interpol_module = interpol_multi.interpolate()
        E = interpol_multi.E


    vert_sequence = interpol_module.get_vert_sequence().detach()

    return shape_x, shape_y, vert_sequence, E


def run_faust(i_file, method):
    """Compute an interpolation for a single pair of faust"""

    param = get_param(method)
    shape_x, shape_y = load_faust_pair(i_file)

    return run_single(shape_x, shape_y, method)


def main_faust(method, i_file):
    """Main script to compute an interpolation for a pair of faust"""

    print("Run method ", method, " for FAUST ", str(i_file))

    shape_x, shape_y, vert_sequence, E = run_faust(i_file, method)

    print("Finished method ", method, " for FAUST ", str(i_file))

    plot_interpolation(vert_sequence, shape_x, shape_y)


def main_dataset(method, dataset, folder_idx=None):
    """Run a complete dataset and save the results for all pairs"""

    global log
    log = False

    if folder_idx is None:
        folder_idx = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        folder_idx = str(folder_idx).zfill(3)

    folder_name = data_folder_out + method + "/" + dataset + "_" + folder_idx + "/"

    if dataset == "FAUST":
        run_handle = run_faust
        num_files = 90

    file_array = np.random.permutation(num_files)

    for i in range(num_files):
        try:
            i_file = file_array[i]

            file_name = "seq_" + str(i_file).zfill(3) + ".mat"

            if os.path.isfile(folder_name + file_name):
                print('File ' + file_name + ' already exists.')
                continue

            print('Computing file ' + file_name + ' with ' + method)

            start_time = datetime.datetime.now()

            shape_x, shape_y, vert_sequence, E = run_handle(i_file, method, reverse_pairs=False)

            print(dataset, " ", i_file, " with ", method, " produced error: ", E)

            time_elapsed = datetime.datetime.now() - start_time
            time_elapsed = '{}'.format(time_elapsed)
            print('Time elapsed (hh:mm:ss.ms) ' + time_elapsed)

            save_sequence(folder_name, file_name, vert_sequence, shape_x, shape_y, time_elapsed)
        except Exception as e:
            print("Expection in file " + str(i_file))
            print(type(e))
            print(e.args)
            print(e)


if __name__ == "__main__":
    #choose method
    method = "ham"
    # method = "div"
    # method = "extra"
    
    #choose FAUST pair
    i_file = 46
    
    #run FAUST on a subsampled version due to high resolution
    main_faust(method, i_file)

