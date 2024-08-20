"""
@Name: Evaluator.py
@Author: Zeyu Zhang
@Date: 2024/4/28-15:59

Script: This is the class of evaluator. The function of evaluation is to control the process.
"""

import importlib
import os
from TimeFlow import TimeFlow
from methods.FullMemAgent import FullMemAgent
from ResultCollector import ResultCollector,SubResultCollector, SingleRepeatCollector
from utils import load_test_set


class Evaluator():
    """
    The evaluation may include many evaluations (combination of different dataset and models).
    """
    def __init__(self, config):
        self.config = config
        self.meta_config()

        self.result_collector = ResultCollector(config)

    def meta_config(self):
        if 'cuda' in self.config['meta_config']:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.config['meta_config']['cuda']

    def eval(self):
        data_list = self.config['evaluation_config']['data_list']
        model_list = self.config['evaluation_config']['model_list']
        for single_data in data_list:
            data_name, data_path = single_data['data_name'], single_data['data_path']
            for QAType in self.config['evaluation_config']['QAType_list']:
                for single_model in model_list:
                    model_name, model_path, model_config = single_model['model_name'], single_model['model_path'], single_model['model_config']
                    subevaluator = SubEvaluator(self.config, data_path, model_path, model_config, data_name, model_name, QAType)
                    sub_result_collector = subevaluator.eval()
                    sub_result_collector.statistic() # Only save
                    # print(sub_result_collector.statistic())
                    self.result_collector.add(data_name,model_name,sub_result_collector)

    def get_result(self):
        return self.result_collector

class SubEvaluator():
    """
    Evaluate a single dataset with a single model.
    """
    def __init__(self, config, data_path, model_path, model_config, data_name, model_name, QAType):
        self.config = config
        self.data_path = data_path
        self.model_path = model_path
        self.data_name = data_name
        self.model_name = model_name
        self.QAType = QAType

        self.agent = self.create_model(model_path, model_config)

    def create_model(self,model_path, model_config):
        module_name, class_name = model_path.rsplit('.',1)
        module = importlib.import_module(module_name)
        cls = getattr(module,class_name)
        model = cls(model_config)
        return model

    def eval(self, save_result = True):
        sub_result_collector = SubResultCollector(self.config, self.data_name, self.model_name, self.QAType)

        repeat_time = 10
        test_data = load_test_set(self.data_path, self.QAType, repeat_time)

        for index in range(repeat_time):
            print('--- Repeat %d ---' % index)
            print('[%s - %s - %s]' % (self.data_path, self.model_path, self.QAType))

            single_repeat_collector = SingleRepeatCollector()

            for jndex, traj in enumerate(test_data[index]):
                self.agent.reset()
                
                timeflow = TimeFlow(self.config,self.agent,traj)
                
                res = timeflow.run()
                print('%d Finish!' % jndex, res)
            
                single_repeat_collector.add(res)

            sub_result_collector.add(single_repeat_collector.get_multi_repeat_res())

        return sub_result_collector
