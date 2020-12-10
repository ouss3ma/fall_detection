import os
import sys
import argparse
import linecache

# input: label file
# output: file contain label of presence/non presence of action

windows = 15 #number frames of a subsequence
threshold = 2

def read_skeleton(file_in,file_out):
    print(file_in)
    fin = open(file_in, 'r')
    Lines = fin.readlines()
    action = []
    t=0
    for line in Lines:

        line=line.split(',')
        for i in range(t , int(line[1])):
            action += [0]

        for i in range(int(line[1]) , int(line[2])):
            action += [1]

        t = int(line[2])

        #if file_in == '/home/oussema/code/PKU-MMD/Train_Label_PKU_final/train/0043-L.txt':
         #   print(t)

    action = action[:(len(action)//windows)*windows]
    


    sub_action=[]
    count = 0
    for i in range(0,len(action),windows):


        if sum(action[i: windows + i]) == 0:
            sub_action += [0]
        elif sum(action[i : windows+i]) < threshold:
            sub_action += [0]

        else:
            sub_action += [1]

    print(count)

    fout = open(file_out, 'w')
    for i in sub_action:
        fout.writelines(str(i))
    fout.close()




def gendata(data_path, out_folder):
    for filename in os.listdir(data_path):
        read_skeleton(os.path.join(data_path,filename),os.path.join(out_folder,filename))





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PKU-MMD Data Converter.')
    parser.add_argument('--data_path',
                        default='/home/oussema/code/PKU-MMD/Train_Label_PKU_final/test')
    
    parser.add_argument('--out_folder', default='/home/oussema/code/st-gcn/pku/pku_action_detect/test')
    
    arg = parser.parse_args()


    if not os.path.exists(arg.out_folder):
        os.makedirs(arg.out_folder)

    gendata(arg.data_path,arg.out_folder)
