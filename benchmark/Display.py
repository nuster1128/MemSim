"""
@Name: Display.py
@Author: Zeyu Zhang
@Date: 2024/4/28-16:09

Script: This is the class to display the result from evaluation.
"""
import os
import pandas as pd
from prettytable import PrettyTable
import numpy as np
from utils import load_json, save_json

NAME_CONVERT = {
    'accuracy': 'Accuracy',
    'recall': 'Recall@5',
    'write_time': 'Write Time(s)', 'process_time': 'Process Time(s)', 'read_time': 'Read Time(s)'
}



class Display():
    def __init__(self, config, result_collector):
        self.config = config
        self.result_collector = result_collector

    def table_show(self):
        result_all = ''
        model_list = self.result_collector.get_model_keys()
        for dataset_name in self.result_collector.get_dataset_keys():
            dataset = self.result_collector.get_single_dataset(dataset_name)
            table_data = {m:[] for m in self.config['evaluation_config']['metrics']}
            for model_name in model_list:
                _, sta = dataset[model_name].statistic()
                for k, v in sta.items():
                    table_data[k].append('%.2f±%.2f' % (v[0],v[1]))

            df = pd.DataFrame(table_data,index=model_list)
            pretty_table = convert_dataframe_to_prettytable(df, dataset_name)
            result_all += pretty_table.__str__()
            print(pretty_table)
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write(result_all)
                

def convert_dataframe_to_prettytable(df, title):
    table = PrettyTable()
    table.title = title
    table.field_names =['Models'] + [NAME_CONVERT[n] for n in df.columns.tolist()]
    for row in df.itertuples(index=True):
        table.add_row(row)
    return table