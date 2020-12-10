import os
import sys
import pickle
import linecache
import argparse
import numpy as np
from numpy.lib.format import open_memmap

#training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
#training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
training_subjects=[]
#training_cameras = [2, 3]
training_cameras = []
max_body = 1
num_joint = 25
max_frame = 30
sequences = 154
windows=30

toolbar_width = 30


def read_skeleton(file,s):

    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = windows

        skeleton_sequence['frameInfo'] = []

        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = 1
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}

                body_info['numJoint'] = 25
                body_info['jointInfo'] = []
                theline = linecache.getline(file, s*windows + t+1)
                #print(s*30 + t+1)
                line=theline.split()
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z']
                    joint_info = {
                        k: float(v)

                        for k, v in zip(joint_info_key,
                                        line[v*3:v*3+3])

                    }
                    #print(joint_info)
                    #print (s*30+t)
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, s , max_body=2, num_joint=25):

    seq_info = read_skeleton(file,s)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:

                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    #for filename in os.listdir(data_path) :
    for fames in range (1,sequences):
        filename = '0002-L.txt'
        if filename in ignored_samples:
            continue


        action_class = 0
        subject_id = 0
        camera_id = 0

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()




        if issample:
            if action_class == 11:
                sample_name.append(fames)
                #sample_label.append(action_class - 1)
                sample_label.append(1)
            else:
                sample_name.append(fames)
                # sample_label.append(action_class - 1)
                sample_label.append(0)


    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap('{}/{}_data.npy'.format(out_path, part),
                     dtype='float32',
                     mode='w+',
                     shape=(len(sample_label), 3, max_frame, num_joint,
                            max_body))

    for i, s in enumerate(sample_name):

        print_toolbar(
            i * 1.0 / len(sample_label),
            '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                i + 1, len(sample_name), benchmark, part))
        #print (s)
        data = read_xyz(os.path.join(data_path, '0002-L.txt'),s,
                        max_body=max_body,
                        num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PKU-MMD Data Converter.')
    parser.add_argument('--data_path',
                        default='/home/oussema/code/echantillon')
    parser.add_argument(
        '--ignored_sample_path',
        default=None
        #'deprecated/tools/data_processing/nturgbd_samples_with_missing_skeletons.txt'
    )
    parser.add_argument('--out_folder', default='/home/oussema/code/st-gcn/data/PKU')

    benchmark = ['xsub']
    #part = ['train', 'val']
    part = [ 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(arg.data_path,
                    out_path,
                    arg.ignored_sample_path,
                    benchmark=b,
                    part=p)
