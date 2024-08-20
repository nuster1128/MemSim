from utils import create_LLM, remove_space_and_ent
from methods.BaseAgent import BaseAgent
import numpy as np

class OracleMemAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.llm = create_LLM(config['LLM_config'])
        self.memory_list = []
        
    def __read_memory__(self):
        mem = ''
        for index, m in enumerate(self.memory_list):
            mem += '[%d] %s\n' % (index, m)
        return mem

    def reset(self, **kwargs):
        self.memory_list = []

    def observe_without_action(self, obs):
        self.memory_list.append(obs)

    def response_answer(self, question, choices, time):
        prompt = '[用户消息] \n'
        prompt += self.__read_memory__()
        prompt += '[单选题] %s \n' % question
        for k,v in choices.items():
            prompt += '%s: %s\n' % (k,v)
        prompt += '当前时间是 %s\n' % time
        prompt += '请根据[用户消息]，给出[单选题]的正确答案。\n'
        prompt += '只输出答案所对应的选项，不要输出解释或其他任何的内容。\n'
        prompt += '输出样例: A'

        res = remove_space_and_ent(self.llm.fast_run(prompt))

        # print(prompt)
        # print(res)

        return res

    def response_retri(self, question, topk = 5):
        demo_list = [i for i in range(topk)]

        prompt = '[用户消息] \n'
        prompt += self.__read_memory__()
        prompt += '[题目] %s \n' % question
        prompt += '请检索与[题目]最相关的%d条不同的[用户消息]，并以python列表给出对应的编号。\n' % topk
        prompt += '只输出一个python列表，不要输出解释或其他任何的内容。\n'
        prompt += '输出样例: %s' % str(demo_list)

        res = remove_space_and_ent(self.llm.fast_run(prompt))

        try:
            res_list = eval(res)
        except Exception as e:
            return [-1 for i in range(topk)]
        if type(res_list) is not list or len(res_list) > topk:
            return demo_list

        return res_list
    
    def process(self):
        pass

