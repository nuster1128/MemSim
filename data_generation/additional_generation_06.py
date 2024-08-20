import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
import string
from common import rewrite_message, rewrite_question, formulate_QA_additional_judge, llm

trajectory_per_graph = 1 # 10

def rewrite_question_noise(noise, question):
    noise_adj_list = [
        "哎呀，实际上我想问的是：",
        "实际上，我真正的问题是：",
        "等等，我真正想问的问题是：",
        "不对，我真正想要了解的是：",
        "搞错了，我实际想问的问题是：",
        "不好意思，我真正想问的是：",
        "哎，我真正想弄清楚的是，",
        "额，实际上我那个问题是这个，",
        "停一下，我真正想问的是，",
        "哦豁，我其实是想搞清楚，",
        "抱歉哈，我真正想问的是这个，",
        "对了，我想问，",
        "我本来想说的是，",
        "等等，"
        "诶，等等，"
    ]

    noise_adj = np.random.choice(noise_adj_list, size=1, replace=False)[0]

    return "%s%s%s" % (noise, noise_adj, question)

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

        prompt = "[上下文] %s\n" % question
        prompt += "请你随机生成一个的碎碎念，但不要包含上下文中的具体信息。只输出该碎碎念，不要输出其他任何描述信息。\n"
        prompt += "示例输出："
        prompt += "我今天去见了一个客户，他好像才25岁，他的职业我记不清了。我的父亲的生日是什么时候来着？"

        noise = remove_space_and_ent(llm.fast_run(prompt))
        question = rewrite_question_noise(noise, question)
        print(question)

        question, choices, groud_truth = formulate_QA_additional_judge(question, answer)

        question_list.append({
                'qid': 0,
                'question': question,
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

        prompt = "[上下文] %s\n" % question
        prompt += "请你随机生成一个的碎碎念，但不要包含上下文中的具体信息。只输出该碎碎念，不要输出其他任何描述信息。\n"
        prompt += "示例输出："
        prompt += "我今天去见了一个客户，他好像才25岁，他的职业我记不清了。我的父亲的生日是什么时候来着？"

        noise = remove_space_and_ent(llm.fast_run(prompt))
        question = rewrite_question_noise(noise, question)
        print(question)

        question, choices, groud_truth = formulate_QA_additional_judge(question, answer)

        question_list.append({
                'qid': 0,
                'question': question,
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

    output_path_item = '06_noisy_items.json'
    output_path_place = '06_noisy_places.json'
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