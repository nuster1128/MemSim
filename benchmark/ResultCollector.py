"""
@Name: ResultCollector.py
@Author: Zeyu Zhang
@Date: 2024/4/28-20:52

Script: This is the class of result collector.
"""
import numpy as np
import os
from utils import load_json,save_json


class SingleRepeatCollector():
    def __init__(self):
        self.result_list = []
    
    def add(self, data):
        self.result_list.append(data)
    
    def get_multi_repeat_res(self):
        raw = {}
        for point_result in self.result_list:
            for k,v in point_result.items():
                if k not in raw:
                    raw[k] = []
                raw[k].append(v)
        
        sta = {k: np.mean(v) for k,v in raw.items()}
        return sta


class SubResultCollector():
    """
    For a single dataset and a single model.
    """
    def __init__(self, config, data_name, model_name, QAType):
        self.config = config
        self.result_list = []
        
        self.data_name = data_name
        self.model_name = model_name
        self.QAType = QAType

    def add(self, result):
        self.result_list.append(result)

    def __str__(self):
        return self.result_list.__str__()

    def statistic(self):
        raw = {}
        for point_result in self.result_list:
            for k,v in point_result.items():
                if k not in raw:
                    raw[k] = []
                raw[k].append(v)
        
        sta = {k: (np.mean(v), np.std(v)) for k,v in raw.items()}

        result_path = self.config['meta_config']['result_path']
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        res_dir = '%s/%s_%s' % (result_path, self.data_name, self.model_name)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        save_json(raw, '%s/%s_%s_%s.json' % (res_dir,self.data_name,self.QAType,self.model_name))

        return raw, sta
    
class ResultCollector():
    """
    For all the dataset and models.
    """
    def __init__(self, config):
        self.config = config

        self.sub_result_collector_dict = {}

    def add(self, data_name, model_name, sub_result_collector):
        if data_name not in self.sub_result_collector_dict:
            self.sub_result_collector_dict[data_name] = {}
        self.sub_result_collector_dict[data_name][model_name] = sub_result_collector

    def __str__(self):
        return self.sub_result_collector_dict.__str__()

    def get_dataset_keys(self):
        return self.sub_result_collector_dict.keys()

    def get_model_keys(self):
        return self.sub_result_collector_dict[list(self.get_dataset_keys())[0]].keys()

    def get_single_dataset(self, dataset_name):
        return self.sub_result_collector_dict[dataset_name]
