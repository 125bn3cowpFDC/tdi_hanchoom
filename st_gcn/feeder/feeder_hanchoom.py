# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import time

# operation
import st_gcn.feeder.tools


class Feeder_hanchoom(torch.utils.data.Dataset):

    def __init__(self,
                 data_path,
                 label_path,
                 mode,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=1,
                 num_person_out=1,
                 temporal_downsample_step=1,
                 debug=False):
        self.debug = debug
        self.mode = mode
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.temporal_downsample_step = temporal_downsample_step
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        print('SAMPLE_ID = ', sample_id)
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id])
        print('LABEL = ', self.label)
        has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [
                s for h, s in zip(has_skeleton, self.sample_name) if h
            ]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 3  #channel
        self.T = 100  #frame
        self.V = 18  #joint
        self.M = self.num_person_out  #person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):
 
        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            #print('FRAME_INDEX = ', frame_index)
            if frame_index >= self.T:
                break
            for m, skeleton_info in enumerate(frame_info["skeleton"]):  # m = number of person 
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                
                #print('DATA_NUMPY_FIRST: ', data_numpy.shape)
                data_numpy[0, frame_index, :, m] = pose[0::2]
                #print('P1: ', pose[0::2])
                data_numpy[1, frame_index, :, m] = pose[1::2]
                #print('P2: ', pose[1::2])
                data_numpy[2, frame_index, :, m] = score
                #print('P3: ', score)

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # processing
        if self.temporal_downsample_step != 1:
            if self.mode is 'train':
                data_numpy = st_gcn.feeder.tools.downsample(data_numpy,
                                              self.temporal_downsample_step)
            else:
                data_numpy = st_gcn.feeder.tools.temporal_slice(
                    data_numpy, self.temporal_downsample_step)
                    
        # data augmentation
        if self.random_shift:
            data_numpy = st_gcn.feeder.tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = st_gcn.feeder.tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = st_gcn.feeder.tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = st_gcn.feeder.tools.random_move(data_numpy)

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]
        #print('DATA_NUMPY: ', data_numpy.shape)

        # match poses between 2 frames
        if self.pose_matching:
            data_numpy = st_gcn.feeder.tools.openpose_match(data_numpy)

        #print('DATA_NUMPY pose matching: ', data_numpy.shape)
        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def test(data_path, label_path, vid=None, graph=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder_hanchoom(
            data_path,
            label_path,
            mode = 'val',
            pose_matching=False,
            num_person=1),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is None:
        index = 0
    elif type(vid) == int:
        index = vid
    else:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)

    name = loader.dataset.sample_name[index]

    data, label = loader.dataset[index]
    data = data.reshape(data.shape)

    # for batch_idx, (data, label) in enumerate(loader):
    C, T, V, M = data.shape
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if graph is None:
        p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
        pose = [
            ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
        ]
        ax.axis([-1, 1, -1, 1])
        for t in range(T):
            # print t
            for m in range(M):
                pose[m].set_xdata(data[0, t, :, m])
                pose[m].set_ydata(data[1, t, :, m])
            fig.canvas.draw()
            plt.pause(0.001)
            # raw_input(t)
    else:
        p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
        import sys
        from os import path
        sys.path.append(
            path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        G = import_class(graph)()
        edge = G.inward
        pose = []
        for m in range(M):
            a = []
            for i in range(len(edge)):
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
            pose.append(a)
        ax.axis([-1, 1, -1, 1])
        for t in range(T):
            for m in range(M):
                for i, (v1, v2) in enumerate(edge):
                    pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                    pose[m][i].set_ydata(-data[1, t, [v1, v2], m])
            fig.canvas.draw()
            plt.pause(0.001)
            # raw_input(t)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    data_path = './data/hanchoom-skeleton/hanchoom_val'
    label_path = './data/hanchoom-skeleton/hanchoom_val_label.json'
    graph = 'st_gcn.graph.hanchoom'
    # test(data_path, label_path, vid='iqkx0rrCUCo', graph=graph)
    test(data_path, label_path, vid=11111, graph=graph)
    # test(data_path, label_path, vid = 11199, graph=graph)
