"""
@Name: LLM.py
@Author: Zeyu Zhang
@Date: 2024/3/22-16:24

Script:
"""
import json, os
import openai
from openai import OpenAI
from zhipuai import ZhipuAI
import time
import yaml
import re
import random


def create_LLM(llm_config):
    if llm_config['model_name'] == 'glm-4':
        return GLM_LLM(llm_config)
    if llm_config['model_name'] == 'Llama-3-8B-Instruct':
        return Local_LLM(llm_config)
    if llm_config['model_name'] == 'Mistral-7B-Instruct-v0.3':
        return Local_LLM(llm_config)
    if llm_config['model_name'] == 'glm-4-9b-chat':
        return Local_LLM(llm_config)
    
class LLM():
    def __init__(self, config):
        self.config = config
        self.model_name = config['model_name']
        self.model_type = config['model_type']    # Include 'remote' and 'local'.

    def fast_run(self, query):
        raise NotImplementedError

class GPT_LLM(LLM):
    def __init__(self, config):
        super().__init__(config)

        self.client = OpenAI(api_key=self.config['api_key'])

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content
        }

    def run(self, message_list, temperature=1.0, penalty_score=0.0):
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=message_list,
            temperature=temperature,
            frequency_penalty=penalty_score,
            presence_penalty=penalty_score
        )
        response = self.parse_response(response)
        return response

    def fast_run(self, query, temperature=1.0, penalty_score=0.0):
        response = self.run([{"role": "user", "content": query}], temperature, penalty_score)
        return response['result']

class GLM_LLM(LLM):
    def __init__(self, config):
        super().__init__(config)

        self.client = ZhipuAI(api_key=self.config['api_key'])

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content
        }

    def run(self, message_list, temperature=0.95):
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=message_list,
            temperature=temperature,
        )
        response = self.parse_response(response)
        return response

    def fast_run(self, query, temperature=0.95):
        for i in range(5):
            try:
                response = self.run([{"role": "user", "content": query}], temperature)
                return response['result']
            except Exception as e:
                print(e)

class Local_LLM(LLM):
    def __init__(self, config):
        super().__init__(config)
    
        self.server_port = config['server_port']
        self.client = OpenAI(api_key='none',base_url="http://localhost:%d/v1" % self.server_port)

    def run(self, message_list, temperature=0.95):
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=message_list,
            temperature=temperature,
        )
        response = self.parse_response(response)
        return response
    
    def parse_response(self, response):
         return {
            'result': response.res_message
        }

    def fast_run(self, query, temperature=0.95):
        response = self.run([{"role": "user", "content": query}], temperature)
        return response['result']

# def trace_gpu():
#     total_usage = 0.0
#     for i in range(py3nvml.nvmlDeviceGetCount()):
#         handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
#         mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
#         total_usage += mem_info.used
#     total_usage /= 1024 * 1024
#     return total_usage  # MB

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path,'w', encoding='utf-8') as f:
        json.dump(data,f, indent=4,ensure_ascii=False)

def load_config(path):
    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

def remove_space_and_ent(s):
    return s.replace(" ", "").replace("\n", "")

def load_test_set(data_path, QAType, repeat_time):
    test_set = []

    all_test_set = []
    data = load_json(data_path)[QAType]
    for scenario, scenario_data in data.items():
        all_test_set += scenario_data
    # all_test_set = all_test_set[:20]  ## MUST DELETE FOR NON-DEBUD TIME
    random.shuffle(all_test_set)

    average_num = int(len(all_test_set)/repeat_time)
    for i in range(repeat_time):
        test_set.append(all_test_set[i*average_num:(i+1)*average_num])
    
    return test_set
