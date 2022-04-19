"""
    This script divides the room file point clouds into train, eval, and test
    sets by providing a portion of each area into each of the above sets.

    Author: Ali Ghelmani,       Date: Nov. 30, 2021
"""
import numpy as np
import math

train_ratio = 0.6
eval_ratio = 0.2

def area_data_distributor(area_name, full_data_info):
    """
        This function gets as input an area name and the full data list
        and outputs the train, eval, and test lists for the given area.
    """

    area_data = [f"{area.strip()}: {index}\n" for index, area in enumerate(full_data_info) if area_name in area]
    perm_area = np.random.permutation(np.array(area_data))

    eval_idx = math.floor(train_ratio * len(area_data)) + 1
    test_idx = math.floor((train_ratio + eval_ratio) * len(area_data)) + 1

    return list(perm_area[:eval_idx]), list(perm_area[eval_idx:test_idx]), list(perm_area[test_idx:])



if __name__ == "__main__":

    with open("./bridge_npy_hdf5_data/room_filelist.txt", 'r') as f:
        all_data = f.readlines()

    train_list = []
    eval_list = []
    test_list = []
    areas= ["Area_1_", "Area_2_", "Area_3_", "Area_4_", "Area_5_"]

    for area in areas:
        train, eval, test = area_data_distributor(area, all_data)

        train_list.extend(train)
        eval_list.extend(eval)
        test_list.extend(test)
    
    root_dir = "./bridge_npy_hdf5_data/room_filelist_"
    
    with open(root_dir + "train.txt", 'w') as f:
        f.writelines(train_list)
    
    with open(root_dir + "eval.txt", 'w') as f:
        f.writelines(eval_list)
    
    with open(root_dir + "test.txt", 'w') as f:
        f.writelines(test_list)