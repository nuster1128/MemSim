"""
@Name: TimeFlow.py
@Author: Zeyu Zhang
@Date: 2024/4/28-15:49

Script: This is the class of time flow.
"""
import json
import time
import numpy as np
from utils import remove_space_and_ent
from methods.OracleMemAgent import OracleMemAgent
from methods.NoiseMemAgent import NoiseMemAgent

def get_recall(res, std):
    res = list(set(res))
    std_set = set(std)
    # print(res)
    # print(std_set)
    ct = 0
    for step_id in res:
        if step_id in std:
            ct += 1
    return ct/len(std_set)

def cal_metrics(agent_res, std):
    accuracy = float(agent_res['answer'] == std['ground_truth'])
    recall = get_recall(agent_res['retri'], std['target_step_id'])

    return {
        'accuracy': accuracy,
        'recall': recall,
        'write_time': np.mean(agent_res['write_time_list']),
        'process_time': np.mean(agent_res['process_time_list']),
        'read_time': np.mean(agent_res['read_time'])
    }

class TimeFlow():
    def __init__(self, config, agent, traj):
        self.config = config
        self.agent = agent
        self.message_list = traj['message_list']
        self.total_step = len(self.message_list)
        self.QA = traj['QA']

    def run(self):
        target_step_id_list = self.QA['target_step_id']
        write_time_list, proces_time_list = [], []
        for step, message in enumerate(self.message_list):
            # print('Current Step %d' % step)

            timestamp_01 = time.perf_counter()
            if self.agent.__class__ == OracleMemAgent:
                if step in target_step_id_list:
                    self.agent.observe_without_action(message)
            elif self.agent.__class__ == NoiseMemAgent:
                if message[-1] != ')':
                    self.agent.observe_without_action(message)
            else:
                self.agent.observe_without_action(message)
            timestamp_02 = time.perf_counter()
            

            # Process.
            self.agent.process()
            timestamp_03 = time.perf_counter()

            write_time_list.append(timestamp_02-timestamp_01)
            proces_time_list.append(timestamp_03-timestamp_02)

        # QA.
        question = self.QA['question']
        choices = self.QA['choices']
        QA_time = self.QA['time']

        timestamp_04 = time.perf_counter()
        answer_res = self.agent.response_answer(question, choices, QA_time)
        timestamp_05 = time.perf_counter()

        retri_res = self.agent.response_retri(question)
        read_time = timestamp_05 - timestamp_04

        agent_res = {
            'answer': answer_res,
            'retri': retri_res,
            'write_time_list': write_time_list,
            'process_time_list': proces_time_list,
            'read_time': read_time
        }

        result = cal_metrics(agent_res, self.QA)
        return result
