"""
@Name: LLM.py
@Author: Zeyu Zhang
@Date: 2024/3/22-16:24

Script:
"""
import json
from openai import OpenAI
from zhipuai import ZhipuAI
import yaml
from datetime import datetime, timedelta
import numpy as np
from copy import deepcopy

def create_LLM(llm_config):
    if llm_config['model_name'] == 'GLM-4-0520':
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
                return '[无效输出]'

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

def remove_space_and_ent(s):
    return s.replace(" ", "").replace("\n", "")


NUM_TO_WEEKDAY = {
    1: '周一',
    2: '周二',
    3: '周三',
    4: '周四',
    5: '周五',
    6: '周六',
    7: '周日'
}

WEEKDAY_TO_NUM = {
    '一': 1,
    '二': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '日': 7
}

NUM_TO_HOUR = {
    6: '早上六点',
    7: '早上七点',
    8: '早上八点',
    9: '上午九点',
    10: '上午十点',
    11: '中午十一点',
    12: '中午十二点',
    13: '下午一点',
    14: '下午两点',
    15: '下午三点',
    16: '下午四点',
    17: '下午五点',
    18: '下午六点',
    19: '晚上七点',
    20: '晚上八点',
    21: '晚上九点',
    22: '晚上十点',
}

class TimeClock():
    def __init__(self) -> None:

        self.mean, self.var = 10, 5

        self.start_date = (2024, 4, 1)
        self.start_date = datetime(self.start_date[0], self.start_date[1], self.start_date[2])

        self.current_time = {
            'week': 1,
            'day': 1,
            'hour': 8,
            'total_days': 0
        }

    def get_current_timestamp(self):
        return self.start_date + timedelta(days=self.current_time['total_days'], hours=self.current_time['hour'],minutes=np.random.randint(0,60))


    def get_current_time(self):
        formatted_date = (self.start_date + timedelta(days=self.current_time['total_days'])).strftime("%Y年%m月%d日")

        return '%s %s %02d:%02d' % (formatted_date, NUM_TO_WEEKDAY[self.current_time['day']],self.current_time['hour'],np.random.randint(0,60))

    def update_time(self):
        gap = min(int(np.random.normal(self.mean,self.var))+1,23)
        self.current_time['hour']+=gap
        while self.current_time['hour'] >=24:
            self.current_time['hour']-=24
            self.current_time['day']+=1
            self.current_time['total_days'] += 1
        while self.current_time['day'] > 7:
            self.current_time['day'] -=7
            self.current_time['week'] +=1

        if self.current_time['hour'] >= 22 :
            self.current_time['hour'] = 7
            self.current_time['day'] += 1
            self.current_time['total_days'] += 1
            if self.current_time['day'] > 7:
                self.current_time['day'] -= 7
                self.current_time['week'] += 1

        if self.current_time['hour'] <= 7 :
            self.current_time['hour'] = 7
    
    def reltime_to_abstime(self, base_abstime, rel_gap):
        # base_abstime + rel_gap --> target_abstime
        day_gap, target_date, day_time = rel_gap[:-5], rel_gap[-5], rel_gap[-4:]
        if day_gap == '下周':
            days_until_next_monday = 7 - base_abstime.weekday() - 1 + WEEKDAY_TO_NUM[target_date]
            next_monday = base_abstime + timedelta(days=days_until_next_monday)
        elif day_gap == '下下周':
            days_until_next_monday = 7 - base_abstime.weekday() - 1 + 7 + WEEKDAY_TO_NUM[target_date]
            next_monday = base_abstime + timedelta(days=days_until_next_monday)
        else:
            raise "Error for rel-abs time transfer."

        if day_time == '上午九点':
            target_time = next_monday.replace(hour=9, minute=0, second=0, microsecond=0)
            h = 9
        elif day_time == '下午两点':
            target_time = next_monday.replace(hour=14, minute=0, second=0, microsecond=0)
            h = 14
        elif day_time == '晚上七点':
            target_time = next_monday.replace(hour=16, minute=0, second=0, microsecond=0)
            h = 19
        else:
            raise "Error for rel-abs time transfer."
        
        formatted_date = target_time.strftime("%Y年%m月%d日")

        return '%s %s %02d:%02d' % (formatted_date, NUM_TO_WEEKDAY[target_time.weekday()+1],h,0)
    
    def format_time_to_timestamp(self, format_time):
        return datetime.strptime(format_time[:len('2024年04月03日')] + format_time[-len('10:49'):], "%Y年%m月%d日%H:%M")

    def calculate_reltime(self, base_abstime, target_abstime):
        target_abstime = datetime.strptime(target_abstime, "%Y-%m-%d %H:%M")

        # target_abstime - base_abstime = rel_gap
        weekday, hour = target_abstime.weekday() + 1, target_abstime.hour
        text_weekday = NUM_TO_WEEKDAY[weekday]
        text_hour = NUM_TO_HOUR[hour]

        delta = target_abstime - base_abstime
        past_week = int((base_abstime.weekday() + delta.days)/7)

        new_rel_week = ''
        while past_week != 0:
            new_rel_week += '下'
            past_week -= 1

        return '%s周%s%s' % (new_rel_week, text_weekday, text_hour)
        

    def refine_rel_time(self, abs_pre, rel_pre, abs_cur):
        abs_pre = self.format_time_to_timestamp(abs_pre)
        pre_day, pre_weekday = abs_pre.day, abs_pre.weekday()
        cur_day, cur_weekday = abs_cur.day, abs_cur.weekday()
        past_week = int(((cur_day - cur_weekday) - (pre_day - pre_weekday))/7)
        if ((cur_day - cur_weekday) - (pre_day - pre_weekday)) % 7 !=0 :
            raise "Bug in TimeClock.refine_rel_time()"

        print("---- EXECUTE! ----")
        print(abs_pre, rel_pre, abs_cur, past_week)

        # abs_pre = self.format_time_to_timestamp(abs_pre)
        # abs_pre = deepcopy(abs_pre).replace(hour=9, minute=0, second=0, microsecond=0)
        # abs_cur = deepcopy(abs_cur).replace(hour=9, minute=0, second=0, microsecond=0)
        # rel_week = rel_pre[:-6]

        # # print(abs_pre.strftime("%Y年%m月%d日"), abs_cur.strftime("%Y年%m月%d日"))
        # delta = abs_cur - abs_pre
        # # print(abs_pre.weekday(),delta.days)
        # past_week = int((abs_pre.weekday() + delta.days)/7)
        # # print(past_week)
        # new_rel_week = '%s' % rel_week
        # # print(new_rel_week)
        new_rel_week = '%s' % rel_pre[:-6]
        while past_week != 0:
            if len(new_rel_week) == 0:
                new_rel_week += '上'
            elif new_rel_week[0] == '下':
                new_rel_week = new_rel_week[:-1]
            elif new_rel_week[0] == '上':
                new_rel_week += '上'
            past_week -= 1
        if len(new_rel_week) == 0:
            new_rel_week += '本'
        # print(new_rel_week)
        rel_cur = new_rel_week + rel_pre[-6:]
        print(rel_cur)
        print('---- End ----')
        # print(rel_cur)
        return rel_cur
