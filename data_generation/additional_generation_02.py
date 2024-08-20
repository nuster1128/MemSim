import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
import string
from common import rewrite_message, rewrite_question, formulate_QA_additional_judge, llm

trajectory_per_graph = 1 # 10

def generate_condition_facts_addition(graph_list):
    B1_attrs_num = 5
    question_num = 1
    def generate_condition_facts_01a_item(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        item = graph['items'][0]
        message_list.append({
            'message': rewrite_message("我%s的%s是%s。" % (item['关系'],item['物品类型'], item['物品名称']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        message_list.append({
            'message': rewrite_message("我认为%s是%s。" % (item['物品名称'], item['物品评价']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        question = "我%s的%s怎么样？" % (item['关系'],item['物品类型'])

        prompt = "请简练地用两个短句总结我的评论：%s\n" % item['物品评价']
        prompt += "只输出总结，不需要重复问题，不要输出其他任何描述信息。"
        prompt += "输出样例：性能强劲，续航待提升"
        answer = remove_space_and_ent(llm.fast_run(prompt))

        noise_message_list = []
        place = graph['places'][0]
        noise_message_list.append({
            'message': rewrite_message("我%s的%s是%s。" % (place['关系'],place['地点类型'], place['地点名称']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        noise_message_list.append({
            'message': rewrite_message("我认为%s是%s。" % (place['地点名称'], place['地点评价']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        question, choices, groud_truth = formulate_QA_additional_judge(question, answer)

        question_list.append({
                'qid': 0,
                'question': rewrite_question(question),
                'answer': answer,
                'target_step_id': [0, 1],
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

    def generate_condition_facts_01a_place(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []


        place = graph['places'][0]
        message_list.append({
            'message': rewrite_message("我%s的%s是%s。" % (place['关系'],place['地点类型'], place['地点名称']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        message_list.append({
            'message': rewrite_message("我认为%s是%s。" % (place['地点名称'], place['地点评价']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        item = graph['items'][0]
        noise_message_list.append({
            'message': rewrite_message("我%s的%s是%s。" % (item['关系'],item['物品类型'], item['物品名称']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        noise_message_list.append({
            'message': rewrite_message("我认为%s是%s。" % (item['物品名称'], item['物品评价']), charact),
            'time': time_clock.get_current_time(),
            'place': graph['user_profile']['工作地']
        })
        time_clock.update_time()

        question = "我%s的%s怎么样？" % (place['关系'],place['地点类型'])

        prompt = "请简练地用两个短句总结我的评论：%s\n" % place['地点评价']
        prompt += "只输出总结，不需要重复问题，不要输出其他任何描述信息。"
        prompt += "输出样例：环境优美，但交通不便"
        answer = remove_space_and_ent(llm.fast_run(prompt))

        question, choices, groud_truth = formulate_QA_additional_judge(question, answer)

        question_list.append({
                'qid': 0,
                'question': rewrite_question(question),
                'answer': answer,
                'target_step_id': [0, 1],
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

    output_path_item = '02_conditional_items.json'
    output_path_place = '02_conditional_places.json'
    data_list_item = []
    data_list_place = []
    for index, graph in enumerate(graph_list):
        print('--- %d Graph ---' % index)
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_condition_facts_01a_item(graph)
            data_list_item.append({
                'tid': len(data_list_item),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list_item)-1, 'Finish!')
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_condition_facts_01a_place(graph)
            data_list_place.append({
                'tid': len(data_list_place),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list_place)-1, 'Finish!')
    
    with open(output_path_item,'w', encoding='utf-8') as f:
        json.dump(data_list_item, f, indent=4,ensure_ascii=False)
    with open(output_path_place,'w', encoding='utf-8') as f:
        json.dump(data_list_place, f, indent=4,ensure_ascii=False)

def generate_memory_and_questions(demo_mode = False):
    profiles_path = 'graphs.json'
    with open(profiles_path,'r', encoding='utf-8') as f:
        graph_list = json.load(f)

    if not demo_mode:
        generate_condition_facts_addition(graph_list)
    else:
        generate_condition_facts_addition(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions(demo_mode=True)