import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class DummyDataset(BaseDataset):
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'],
                            help='how to normalize input comparE feature')
        parser.add_argument('--corpus_name', type=str, default='IEMOCAP', help='which dataset to use')
        return parser



    def __init__(self, opt, set_name=None):
        super().__init__(opt)
        self.norm_method = opt.norm_method
        self.corpus_name = opt.corpus_name
        cvNo = opt.cvNo
        self.label = np.array([0])
        self.A_type = opt.A_type
        with open('feature_path_info.json', 'r') as f:
            data = json.load(f)
        file_path = data['my_feature_path']

        feat_dim_V = 342  # 向量的特征维度
        seq_len_V = 50  # 序列的长度

        feat_dim_L = 1024  # 向量的特征维度
        seq_len_L = 22  # 序列的长度

        zero_vector = np.zeros((seq_len_V, feat_dim_V), dtype=np.float32)
        zero_vector_2 = np.zeros((seq_len_L, feat_dim_L), dtype=np.float32)
        self.file_path = file_path
        self.all_A = h5py.File(os.path.join(file_path), 'r')
        with h5py.File('L_zero_vector.h5', 'w') as f:
            # 创建一个数据集 'zero_vector'，并将全零向量写入
            f.create_dataset('zero_vector', data=zero_vector_2)
        with h5py.File('V_zero_vector.h5', 'w') as f:
            # 创建一个数据集 'zero_vector'，并将全零向量写入
            f.create_dataset('zero_vector_2', data=zero_vector)
        self.num_samples = len(self.all_A)
        self.V_type = opt.V_type
        self.all_V = h5py.File('V_zero_vector.h5', 'r')
        self.L_type = opt.L_type
        self.all_L = h5py.File('L_zero_vector.h5', 'r')
        self.miss_type = ['azz']

        if opt.in_mem:
            self.all_A = self.h5_to_dict(self.all_A)
            self.all_V = self.h5_to_dict(self.all_V)
            self.all_L = self.h5_to_dict(self.all_L)

        self.manual_collate_fn = True

    def __len__(self):
        return len(self.label)

    def calc_mean_std(self):
        utt_ids = [utt_id for utt_id in self.all_A.keys()]
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids])
        _feats = feats.reshape(-1, feats.shape[2])
        mean = np.mean(_feats, axis=0)
        std = np.std(_feats, axis=0)
        std[std == 0.0] = 1.0
        return mean, std

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def __getitem__(self, index):
        miss_type = self.miss_type
        missing_index = 0
        with open('name_info.json', 'r') as f:
            data = json.load(f)
        file_name = data['my_file_name']
        A_feat = torch.from_numpy(self.all_A[file_name][()]).float()
        if self.A_type == 'comparE' or self.A_type == 'comparE_raw':
            A_feat = self.normalize_on_utt(A_feat)
        # process V_feat
        V_feat = torch.from_numpy(self.all_V['zero_vector_2'][()]).float()
        # process L_feat
        L_feat = torch.from_numpy(self.all_L['zero_vector'][()]).float()
        label = torch.tensor(4)  # 假设固定标签为 4
        int2name = file_name

        return {
            'A_feat': A_feat,
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name,
            'missing_index': missing_index,
            'miss_type': miss_type
        }

    def h5_to_dict(self, h5f):
        ret = {}
        for key in h5f.keys():
            ret[key] = h5f[key][()]
        return ret

    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        return {
            'A_feat': A,
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }