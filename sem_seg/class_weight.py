import numpy as np
import math

# labels_dict : {ind_label: count_label}
# mu : parameter to tune

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

# random labels_dict
labels_dict = {0: 247000, 1: 2000000, 2: 25000000}

create_class_weight(labels_dict)
