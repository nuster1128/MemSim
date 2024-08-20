import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
from common import rewrite_message, rewrite_question, llm, formulate_QA, rewrite_message_event, rewrite_message_role
import string


trajectory_per_graph = 2

def generate_aggr_role_04a(graph_list):
    key_features = ['年龄', '身高', '生日', '家乡', '工作地', '教育背景']
    # B1_attrs_num = 5
    question_num = 1
    noise_attrs_nums = 1

    def get_QA_info(whole_message_str,aggr_attr_k):
        prompt = '[用户信息] %s \n' % '\n'.join(whole_message_str)
        prompt += '基于以上用户信息，对%s这一方面，请帮我生成一个计数的问题，以有几个人作为开头，风格是口语化的第一人称。' % aggr_attr_k
        prompt += '只输出所生成的问题，不要输出其他描述信息。\n'
        prompt += '输出示例：有几个人的年龄在35岁及以下？'

        question = remove_space_and_ent(llm.fast_run(prompt))

        prompt = '[题目] %s \n' % question
        prompt += '请将上述题目修改为对单个人的判断问句，例如，将“有几个人的年龄在35岁及以下？”修改为“他的年龄在35岁及以下吗？”\n'
        prompt += '只输出修改后的问题，不要输出其他描述信息。\n'
        prompt += '输出示例：他的年龄在35岁及以下吗？'

        question_single = remove_space_and_ent(llm.fast_run(prompt))
        
        ans_count = 0
        for m in whole_message_str:
            prompt = '[用户信息] %s \n' % m
            prompt += '[题目] %s \n' % question_single
            prompt += '请根据用户信息回答题目。如果是，答案请输出1；如果不是，答案请输出0；如果无法判断，答案请输出2。\n'
            prompt += '请注意题目中“以上”和“及以上”的区别，例如，“35岁以上”是指比35岁大但不包含35岁，而“35岁及以上”则包含35岁。\n'
            prompt += '请注意，如果题目中没有特别说明，上半年指1月~6月，下半年指7~12月。\n'
            prompt += '只输出答案对应的编号，不要输出其他描述和解释信息，也不要输出类似“输出：”的字样。\n'
            prompt += '输出示例：1'

            ans_single = remove_space_and_ent(llm.fast_run(prompt))
            max_try = 0
            while ans_single not in ['0', '1', '2']:
                print("Single Answer Parse Error: %s" % ans_single)
                ans_single = remove_space_and_ent(llm.fast_run(prompt))
                if max_try >= 10:
                    ans_single = 0
            if ans_single == '1':
                ans_count += 1
            if ans_single == '2':
                return None, None
        answer = '%s人' % ans_count
        print(question,answer)

        return question, answer



    def generate_single(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        role_list = graph['relation_profiles'] + graph['colleague_profiles']
        aggr_attrs = np.random.choice(key_features, size=question_num, replace=False)
        for qid, aggr_attr_k in enumerate(aggr_attrs):
            whole_message_str = []
            for role in role_list:
                r = role['关系']
                v = role[aggr_attr_k]
                n = role['姓名']

                text = rewrite_message_role("%s是我的%s，其%s是%s。" % (n,r, aggr_attr_k, v), charact)
                whole_message_str.append(text)
                message_list.append({
                    'rel': r,
                    'name': n,
                    'attr': (aggr_attr_k,v),
                    'message': text,
                    'time': time_clock.get_current_time(),
                    'place': graph['user_profile']['工作地']
                })
                time_clock.update_time()

            noise_attrs_list = np.random.choice(list(role_list[0].keys()),size = 1, replace=False)
            for noise_attr in noise_attrs_list:
                if noise_attr not in aggr_attrs:
                    for role in role_list:
                        r = role['关系']
                        v = role[noise_attr]
                        n = role['姓名']

                        text = rewrite_message_role("%s是我的%s，其%s是%s。" % (n,r, noise_attr, v), charact)
                        # whole_message_str += '- %s\n' % text
                        message_list.append({
                            'rel': r,
                            'name': n,
                            'attr': (noise_attr,v),
                            'message': text,
                            'time': time_clock.get_current_time(),
                            'place': graph['user_profile']['工作地']
                        })
                        time_clock.update_time()
            
            question, answer = get_QA_info(whole_message_str, aggr_attr_k)
            if question is None:
                question, choices, groud_truth = '[ERRORQ]', '[ERRORC]', '[ERRORG]'
            else:
                question, choices, groud_truth = formulate_QA(question, answer)
            question_list.append({
                'qid': qid,
                'question': question,
                'answer': answer,
                'target_step_id': [qid*len(role_list) + k for k in range(len(role_list))],
                'choices': choices,
                'ground_truth': groud_truth,
                'time': time_clock.get_current_time()
            })
            time_clock.update_time()
            

        message_list = [{
            'mid': mid,
            'message': m['message'],
            'time': m['time'],
            'place': m['place']
        } for mid, m in enumerate(message_list)] + [{
            'mid': mid+len(message_list),
            'message': m['message'],
            'time': m['time'],
            'place': m['place']
        } for mid, m in enumerate(noise_message_list)] 
        
        return message_list, question_list

    data_list = []
    output_path = '04_aggregative_roles.json'
    for graph in graph_list:
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_single(graph)
            data_list.append({
                'tid': len(data_list),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list)-1, 'Finish!')
    
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4,ensure_ascii=False)

def generate_aggr_event_04b(graph_list):
    key_features = ['地点', '规模', '持续时间']
    cand_noise_attrs = ['主要内容','时间', '地点', '规模', '持续时间']
    # B1_attrs_num = 5
    question_num = 1
    noise_attrs_nums = 1
    def get_QA_info(whole_message_str, aggr_attr_k):
        prompt = '[活动信息] %s \n' % '\n'.join(whole_message_str)
        prompt += '基于以上活动信息，对%s这一方面，请帮我生成一个计数的问题，以有几个活动作为开头，风格是口语化的第一人称。' % aggr_attr_k
        prompt += '只输出所生成的问题，不要输出其他描述信息。\n'
        prompt += '输出示例：有几个活动的地点在上海？'

        question = remove_space_and_ent(llm.fast_run(prompt))

        prompt = '[题目] %s \n' % question
        prompt += '请将上述题目修改为对单个活动的判断问句，例如，将“有几个活动的地点在上海？”修改为“这个活动的地点在上海吗？”\n'
        prompt += '只输出修改后的问题，不要输出其他描述信息。\n'
        prompt += '输出示例：这个活动的地点在上海吗？'

        question_single = remove_space_and_ent(llm.fast_run(prompt))

        ans_count = 0
        for m in whole_message_str:
            prompt = '[活动信息] %s \n' % m
            prompt += '[题目] %s \n' % question_single
            prompt += '请根据活动信息回答题目。如果是，请输出1；如果不是，答案请输出0；如果无法判断，答案请输出2。\n'
            prompt += '只输出答案对应的编号，不要输出其他描述和解释信息，也不要输出类似“输出：”的字样。'
            prompt += '输出示例：1'

            ans_single = remove_space_and_ent(llm.fast_run(prompt))
            max_try = 0
            while ans_single not in ['0', '1', '2']:
                print("Single Answer Parse Error: %s" % ans_single)
                ans_single = remove_space_and_ent(llm.fast_run(prompt))
                if max_try >= 10:
                    ans_single = 0
            if ans_single == '1':
                ans_count += 1
            if ans_count == '2':
                return None, None
        answer = '%s个' % ans_count
        print(question,answer)
        return question, answer

    def generate_single(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        event_list = graph['work_events'] + graph['rest_events']
        aggr_attrs = np.random.choice(key_features, size=question_num, replace=False)
        for qid, aggr_attr_k in enumerate(aggr_attrs):
            whole_message_str = []
            for event in event_list:
                v = event[aggr_attr_k]
                n = event['事件名称']

                text = rewrite_message_event("%s的%s是%s。" % (n, aggr_attr_k, v), charact)
                whole_message_str.append(text)
                message_list.append({
                    'name': n,
                    'attr': (aggr_attr_k,v),
                    'message': text,
                    'time': time_clock.get_current_time(),
                    'place': graph['user_profile']['工作地']
                })
                time_clock.update_time()
            
            noise_attrs_list = np.random.choice(cand_noise_attrs,size = 1, replace=False)
            for noise_attr in noise_attrs_list:
                if noise_attr not in aggr_attrs:
                    for event in event_list:
                        v = event[noise_attr]
                        n = event['事件名称']

                        text = rewrite_message_event("%s的%s是%s。" % (n, noise_attr, v), charact)
                        # whole_message_str.append(text)
                        message_list.append({
                            'name': n,
                            'attr': (noise_attr,v),
                            'message': text,
                            'time': time_clock.get_current_time(),
                            'place': graph['user_profile']['工作地']
                        })
                        time_clock.update_time()
            
            question, answer = get_QA_info(whole_message_str, aggr_attr_k)
            if question is None:
                question, choices, groud_truth = '[ERRORQ]', '[ERRORC]', '[ERRORG]'
            else:
                question, choices, groud_truth = formulate_QA(question, answer)
            question_list.append({
                'qid': qid,
                'question': question,
                'answer': answer,
                'target_step_id': [qid*len(event_list) + k for k in range(len(event_list))],
                'choices': choices,
                'ground_truth': groud_truth,
                'time': time_clock.get_current_time()
            })
            time_clock.update_time()

        message_list = [{
            'mid': mid,
            'message': m['message'],
            'time': m['time'],
            'place': m['place']
        } for mid, m in enumerate(message_list)] + [{
            'mid': mid+len(message_list),
            'message': m['message'],
            'time': m['time'],
            'place': m['place']
        } for mid, m in enumerate(noise_message_list)] 
        
        return message_list, question_list

    data_list = []
    output_path = '04_aggregative_events.json'
    for graph in graph_list:
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_single(graph)
            data_list.append({
                'tid': len(data_list),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list)-1, 'Finish!')
    
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4,ensure_ascii=False)


def generate_memory_and_questions():
    profiles_path = 'graphs.json'
    with open(profiles_path,'r', encoding='utf-8') as f:
        graph_list = json.load(f)

    generate_aggr_role_04a(graph_list[:50])
    generate_aggr_event_04b(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions()