import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
from common import rewrite_message, rewrite_question, llm, get_choices, formulate_QA, rewrite_message_event
import string

trajectory_per_graph = 2

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



def generate_noise_condition_facts_role_06a(graph_list):
    key_features = ['姓名', '年龄', '身高', '生日', '家乡', '工作地', '教育背景', '职业', '职位', '单位名称', '兴趣爱好', '联系电话', '邮箱地址']
    B1_attrs_num = 5
    question_num = 1
    def generate_single_01a(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        role_list = graph['relation_profiles'] + graph['colleague_profiles']
        role_B1_id = np.random.choice(range(len(role_list)), size=1, replace=False)[0]
        role_B1 = role_list[role_B1_id]
        relation = role_B1['关系']
        attrs = np.random.choice(key_features, size=B1_attrs_num, replace=False)
        while attrs[1] == '姓名':
            attrs = np.random.choice(key_features, size=B1_attrs_num, replace=False)
        for k in attrs:
            v = role_B1[k]
            text = rewrite_message("我的%s的%s是%s。" % (relation, k, v), charact)
            message_list.append({
                'rel': relation,
                'name': role_B1['姓名'],
                'attr': (k,v),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
            time_clock.update_time()

        for qid in range(question_num):
            inx_01, inx_02 = np.random.choice(range(len(message_list)), size=2, replace=False)
            real_attr_01, real_attr_02 = message_list[inx_01], message_list[inx_02]
            question = "那个%s是%s的人，%s是什么？" % (real_attr_01['attr'][0], real_attr_01['attr'][1], real_attr_02['attr'][0])
            answer = real_attr_02['attr'][1]

            for noise_role_id in range(len(role_list)):
                if noise_role_id != role_B1_id:
                    rel, k = role_list[noise_role_id]['关系'], real_attr_02['attr'][0]
                    v = role_list[noise_role_id][k]
                    text = rewrite_message("我%s的%s是%s。" % (rel, k, v), charact)

                    noise_message_list.append({
                        'message': text,
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

            question, choices, groud_truth = formulate_QA(question, answer)
            question_list.append({
                'qid': qid,
                'question': question,
                'answer': answer,
                'target_step_id': [int(inx_01), int(inx_02)],
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
    output_path = '06_noisy_roles.json'
    for graph in graph_list:
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_single_01a(graph)
            data_list.append({
                'tid': len(data_list),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list)-1, 'Finish!')
    
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4,ensure_ascii=False)

def generate_noise_condition_facts_event_06b(graph_list):
    key_features = ['主要内容', '地点', '规模', '持续时间']
    # key_features = ['地点', '时间', '规模']
    B1_attrs_num = 3
    question_num = 1
    def generate_single_01a(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        event_list = graph['work_events'] + graph['rest_events']
        event_C1_id = np.random.choice(range(len(event_list)), size=1, replace=False)[0]
        event_C1 = event_list[event_C1_id]

        text = rewrite_message_event("我将要参加%s。" % (event_C1['事件名称']), charact)
        message_list.append({
                'name': event_C1['事件名称'],
                'attr': ('我要参加的活动',event_C1['事件名称']),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
        time_clock.update_time()

        attrs = np.random.choice(key_features, size=B1_attrs_num, replace=False)
        for k in attrs:
            v = event_C1[k]
            text = rewrite_message_event("%s的%s是%s。" % (event_C1['事件名称'], k, v), charact)
            message_list.append({
                'name': event_C1['事件名称'],
                'attr': (k,v),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
            time_clock.update_time()

        inx_01, inx_02 = np.random.choice(range(len(message_list)), size=2, replace=False)
        real_attr_01, real_attr_02 = message_list[inx_01], message_list[inx_02]
        
        for qid in range(question_num):
            for noise_event_id in range(len(event_list)):
                if noise_event_id != event_C1_id:
                    name, k = event_list[noise_event_id]['事件名称'], real_attr_02['attr'][0]
                    if k == '我要参加的活动':
                        v = event_list[noise_event_id]['事件名称']
                        text = rewrite_message_event("我将要参加%s。" % name, charact)
                    else:
                        v = event_list[noise_event_id][k]
                        text = rewrite_message_event("%s的%s是%s。" % (name, k, v), charact)

                    noise_message_list.append({
                        'message': text,
                        'time': time_clock.get_current_time(),
                        'place': graph['user_profile']['工作地']
                    })
                    time_clock.update_time()

            conditional_ans = real_attr_01['attr'][1]
            if real_attr_01['attr'][0] == '时间' and real_attr_01['attr'][1][0] == '下':
                abs_pre = real_attr_01['time']
                rel_pre = real_attr_01['attr'][1]
                rel_cur = time_clock.refine_rel_time(abs_pre, rel_pre, time_clock.get_current_timestamp())
                conditional_ans = rel_cur

            if real_attr_01['attr'][0] == '主要内容':
                question = "那个%s是\"%s\"的活动，%s是什么？" % (real_attr_01['attr'][0], conditional_ans, real_attr_02['attr'][0])
            else:
                question = "那个%s是%s的活动，%s是什么？" % (real_attr_01['attr'][0], conditional_ans, real_attr_02['attr'][0])
            answer = real_attr_02['attr'][1]

            if real_attr_02['attr'][0] == '时间' and answer[0] == '下':
                answer = time_clock.reltime_to_abstime(time_clock.get_current_timestamp(),answer)
            
            prompt = "[上下文] %s\n" % question
            prompt += "请你随机生成一个和上下文相关的碎碎念，但不要包含上下文中的具体信息。只输出该碎碎念，不要输出其他任何描述信息。\n"
            prompt += "示例输出："
            prompt += "我今天去见了一个客户，他好像才25岁，他的职业我记不清了。我的父亲的生日是什么时候来着？"

            noise = remove_space_and_ent(llm.fast_run(prompt))
            question = rewrite_question_noise(noise, question)

            question, choices, groud_truth = formulate_QA(question, answer)
            question_list.append({
                'qid': qid,
                'question': question,
                'answer': answer,
                'target_step_id': [int(inx_01), int(inx_02)],
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
    output_path = '06_noisy_events.json'
    for graph in graph_list:
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_single_01a(graph)
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

    # generate_noise_condition_facts_role_06a(graph_list[:50])
    generate_noise_condition_facts_event_06b(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions()