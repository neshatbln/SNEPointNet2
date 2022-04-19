import os
import glob
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

data_dir = os.path.join(ROOT_DIR, 'data')
mirrored_dir = os.path.join(data_dir, 'mirrored')
if not os.path.exists(mirrored_dir):
    os.mkdir(mirrored_dir)
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths_.txt'))]


for ann_path in anno_paths:
    print(ann_path)
    elements = ann_path.split('/')
    area_h = os.path.join(mirrored_dir+'/'+elements[-3]+’h’)
    area_v = os.path.join(mirrored_dir+'/'+elements[-3]+’v’)

    if not os.path.exists(area_h):
        os.mkdir(area_h) 
    if not os.path.exists(area_v):
        os.mkdir(area_v)

    part_h = os.path.join(area_h+’/’+elements[-2])
    part_v = os.path.join(area_v+'/'+elements[-2])
    if not os.path.exists(part_h):
        os.mkdir(part_h)
    if not os.path.exists(part_v):
        os.mkdir(part_v)

    output_dir_h = os.path.join(part_h+'/'+'Annotations')
    output_dir_v = os.path.join(part_v+'/'+'Annotations')

    if not os.path.exists(output_dir_h):
        os.mkdir(output_dir_h)
    if not os.path.exists(output_dir_v):
        os.mkdir(output_dir_v)
     input = np.loadtxt(data_dir+'/'+'bridge'+'/'+elements[-3]+'/'+elements[-2]+'/'+elements[-2]+'.txt', dtype=np.float, delimiter=' ')
    num=len(input)
    out_h = out_v= input
    for i in range(num):
        # vertically:
        Out_v[i, (2,8)] = -input[i, (2,8)]
        # horizontally:
        Out_h[i,(0,6)] = -input[i,(0,6)]
        np.savetxt(part_h + '/' + elements[-2] + '.txt', out_h, fmt=’%.3f %.3f %.3f %d %d %d %.3f %.3f %.3f’)
        np.savetxt(part_v + '/' + elements[-2] + '.txt', out_v, fmt='%.3f %.3f %.3f %d %d %d %.3f %.3f %.3f')
    for fv in glob.glob(os.path.join(data_dir, 'bridge', anno_path, '*.txt')):
        out_filename = os.path.basename(fv)
        input = np.loadtxt(fv, dtype=np.float, delimiter=' ')
        print (fv)
        num=len(input)
        print (num)
        out_h = out_v = input
        for i in range(num):
            # horizontally:
            out_h[i, (0,6)] = -input[i, (0,6)]
            # vertically:
            out_v[I,(2,8)] = -input[I,(2,8)]

        np.savetxt(output_dir_h + ‘/’ + out_filename, out_h, fmt=’%.3f %.3f %.3f %d %d %d %.3f %.3f %.3f’)
        np.savetxt(output_dir_v + '/' + out_filename, out_v, fmt='%.3f %.3f %.3f %d %d %d %.3f %.3f %.3f')
