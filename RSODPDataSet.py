import json
import sys
import os
import time
import warnings

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

import numpy as np
import torch

import Config

# Ignore warnings
warnings.filterwarnings('ignore')


class RSODPDataSetEntity(DGLDataset):
    def __init__(self, data_dir: str, sample_list: list, ds_type='train'):
        self.data_dir = data_dir
        self.sample_list = sample_list
        self.ds_type = ds_type

    def process(self):
        pass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Want the (idx)th data from sample_list
        cur_sample_ref: dict = self.sample_list[idx]

        # T -> T+1 = target (G, Q)
        cur_T = cur_sample_ref['T']
        cur_Tp1 = cur_T + 1
        GDVQ_Tp1 = np.load(os.path.join(self.data_dir, str(cur_Tp1), 'GDVQ.npy'), allow_pickle=True).item()
        G_Tp1, D_Tp1, Q_Tp1 = torch.from_numpy(GDVQ_Tp1['G']), torch.from_numpy(GDVQ_Tp1['D']), torch.from_numpy(GDVQ_Tp1['Q'])

        # sample: for each time slot: (fg, bg, gg, V)
        cur_sample_inputs = {}
        for temp_feat in Config.TEMP_FEAT_NAMES:
            temp_feat_sample_inputs = []
            for ts in cur_sample_ref['record'][temp_feat]:
                GDVQ_ts = np.load(os.path.join(self.data_dir, str(ts), 'GDVQ.npy'), allow_pickle=True).item()
                V_ts = torch.from_numpy(GDVQ_ts['V'])
                (fg_ts, bg_ts,), _ = dgl.load_graphs(os.path.join(self.data_dir, str(ts), 'FBGraphs.dgl'))
                (gg_ts,), _ = dgl.load_graphs(os.path.join(self.data_dir, 'GeoGraph.dgl'))
                fg_ts.ndata['v'] = V_ts
                bg_ts.ndata['v'] = V_ts
                gg_ts.ndata['v'] = V_ts
                temp_feat_sample_inputs.append((fg_ts, bg_ts, gg_ts))
            cur_sample_inputs[temp_feat] = temp_feat_sample_inputs

        cur_sample_data = {
            'target_G': G_Tp1,
            'target_D': D_Tp1,
            'query': Q_Tp1,
            'record': cur_sample_inputs
        }
        return cur_sample_data


class RSODPDataSet:
    """
        test set: last two weeks
        training set: the remaining samples apart from test set
        validation set: the last 10% of the training set
    """

    def __init__(self, data_dir: str, his_rec_num=7, time_slot_endurance=1):
        self.data_dir = data_dir
        self.req_info = json.load(open(os.path.join(data_dir, 'req_info.json')))
        self.his_rec_num = his_rec_num  # P
        self.time_slot_num_per_day = 24 / time_slot_endurance  # l

        self.total_sample_list = self.constructSampleList()
        self.total_sample_num = len(self.total_sample_list)

        self.train_list, self.valid_list, self.test_list = self.splitTrainValidTest()
        self.train_set = RSODPDataSetEntity(self.data_dir, sample_list=self.train_list, ds_type='train')
        self.valid_set = RSODPDataSetEntity(self.data_dir, sample_list=self.valid_list, ds_type='valid')
        self.test_set = RSODPDataSetEntity(self.data_dir, sample_list=self.test_list, ds_type='test')

    def constructSampleList(self):
        totalH = self.req_info['totalH']
        total_list = []
        for i in range(totalH):
            cur_ts = i + 1
            # Have T=cur_ts data, predict T+1=cur_ts+1
            # For predicting T+1: (T-lP) is the smallest time slot to be considered
            if (cur_ts - self.time_slot_num_per_day * self.his_rec_num <= 0) or (cur_ts + 1 > totalH):
                # Omit incomplete sample
                continue
            St = [int(cur_ts - pm1) for pm1 in range(self.his_rec_num)]  # Tendency: T + 1 - p, p in [1, P]
            Sp = [int(cur_ts + 1 - self.time_slot_num_per_day * (pm1 + 1)) for pm1 in range(self.his_rec_num)]  # Periodicty: T + 1 - lp, p in [1, P]
            Stpm = [(n - 1) for n in Sp]  # Misc-: T - lp, p in [1, P]
            Stpp = [(n + 1) for n in Sp]  # Misc+: T + 2 - lp, p in [1, P]
            cur_sample = {
                'T': cur_ts,
                'record': {
                    'St': St,
                    'Sp': Sp,
                    'Stpm': Stpm,
                    'Stpp': Stpp
                }
            }
            total_list.append(cur_sample)
        return total_list

    def splitTrainValidTest(self):
        # test set: last 2 weeks = 14 days = 14 * l time slots
        first_test_sample_idx = int(-14 * self.time_slot_num_per_day)
        test_set = self.total_sample_list[first_test_sample_idx:]
        # training set: the remaining samples apart from test set
        train_set = self.total_sample_list[:first_test_sample_idx]
        # validation set: the last 10% of the training set
        valid_set_num = int(len(train_set) * 0.1)
        valid_set = train_set[-valid_set_num:]

        return train_set, valid_set, test_set


def testSamplingSpeed(dataset: DGLDataset, batch_size: int, shuffle: bool, tag: str, num_workers=4):
    """ Test the sampling functionality & efficiency """
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    time0 = time.time()
    for i, batch in enumerate(dataloader):
        record, query, target = batch['record'], batch['query'], batch['target']
        sys.stdout.write('\r{} Set - Batch No. {}/{} with time used(s): {}'.format(tag, i+1, len(dataloader), time.time() - time0))
        sys.stdout.flush()
        if i == 0:
            test = batch  # For debugging, set a breakpoint here
    sys.stdout.write('\n')


if __name__ == '__main__':
    path = 'data/ny2016_0101to0331/'
    ds = RSODPDataSet(data_dir=path)
    testSamplingSpeed(ds.train_set, batch_size=20, shuffle=True, tag='Training', num_workers=4)
    testSamplingSpeed(ds.valid_set, batch_size=20, shuffle=False, tag='Validation', num_workers=4)
    testSamplingSpeed(ds.test_set, batch_size=20, shuffle=False, tag='Test', num_workers=4)
