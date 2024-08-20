import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
from common import rewrite_message, rewrite_question, llm, get_choices, rewrite_message_event, rewrite_message_role
import string

def formulate_QA(question, answer, nameB1, nameB2):
    if answer == '无法判断':
        other_answers = [nameB1, nameB2, '都不正确']
    elif answer == nameB1:
        other_answers = [nameB2, '两者一样', '都不正确']
    elif answer == nameB2:
        other_answers = [nameB1, '两者一样', '都不正确']
    elif answer == '两者一样':
        other_answers = [nameB1, nameB2, '无法判断']
    else:
        other_answers = ['两者一样', '都不正确', '钝角']

    groud_truth,choices = get_choices(answer,other_answers)
    return question, choices, groud_truth


trajectory_per_graph = 2

def generate_compare_role_03a(graph_list):
    key_features = ['年龄', '身高', '生日', '教育背景']
    # B1_attrs_num = 5
    question_num = 1
    def generate_single_01a(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        role_list = graph['relation_profiles'] + graph['colleague_profiles']
        role_B_ids = np.random.choice(range(len(role_list)), size=2, replace=False)
        role_B1, role_B2 = role_list[role_B_ids[0]], role_list[role_B_ids[1]]
        relationB1, relationB2 = role_B1['关系'], role_B2['关系']

        compare_attrs = np.random.choice(key_features, size=question_num, replace=False)
        for compare_attr_k in compare_attrs:
            vB1, vB2 = role_B1[compare_attr_k], role_B2[compare_attr_k]

            text = rewrite_message_role("%s是我的%s，其%s是%s。" % (role_B1['姓名'],relationB1, compare_attr_k, vB1), charact)
            message_list.append({
                'rel': relationB1,
                'name': role_B1['姓名'],
                'attr': (compare_attr_k,vB1),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
            time_clock.update_time()

            text = rewrite_message_role("%s是我的%s，其%s是%s。" % (role_B2['姓名'],relationB2, compare_attr_k, vB2), charact)
            message_list.append({
                'rel': relationB2,
                'name': role_B2['姓名'],
                'attr': (compare_attr_k,vB2),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
            time_clock.update_time()

        for qid in range(question_num):
            compare_attr_k = compare_attrs[qid]
            vB1, vB2 = role_B1[compare_attr_k], role_B2[compare_attr_k]
            nameB1, nameB2 = role_B1['姓名'], role_B2['姓名']

            for noise_role_id in range(len(role_list)):
                if noise_role_id not in role_B_ids:
                    v = role_list[noise_role_id][compare_attr_k]
                    r = role_list[noise_role_id]['关系']
                    n = role_list[noise_role_id]['姓名']

                    text = rewrite_message_role("%s是我的%s，其%s是%s。" % (n,r, compare_attr_k, v), charact)
                    message_list.append({
                        'rel': r,
                        'name': n,
                        'attr': (compare_attr_k,v),
                        'message': text,
                        'time': time_clock.get_current_time(),
                        'place': graph['user_profile']['工作地']
                    })
                    time_clock.update_time()


            prompt = '[信息] %s的%s是%s；%s的%s是%s。\n' % (nameB1,compare_attr_k, vB1, nameB2,compare_attr_k,vB2)
            prompt += '基于给出的两人的信息，请帮我生成一个可以回答的，对%s进行比较的问题。' % compare_attr_k
            prompt += '问题的答案应该是%s、%s或两者一样。\n' % (nameB1,nameB2)
            prompt += '只输出所生成问题，不要输出答案，不要输出其他描述信息。\n'
            prompt += '输出示例：张三和李四谁的职位更高？'
            question = remove_space_and_ent(llm.fast_run(prompt))
        
            prompt = '[信息] %s的%s是%s；%s的%s是%s。\n' % (nameB1,compare_attr_k, vB1, nameB2,compare_attr_k,vB2)
            prompt += '基于给出的两人的信息，请帮我回答问题：%s。\n' % question
            prompt += '如果无法根据信息回答问题，请输出 无法判断；如果两者一样，请输出 两者一样；否则，答案应该是 %s 或 %s。\n' % (nameB1, nameB2)
            prompt += '只输出该问题的答案，不要输出其他描述信息。'
            prompt += '输出示例：无法判断'

            ans_pre = remove_space_and_ent(llm.fast_run(prompt))
            ans_ex = remove_space_and_ent(llm.fast_run('RandomSeed(%d)\n%s' % (np.random.randint(1,100),prompt)))
            max_try = 0
            while ans_ex != ans_pre:
                ans_pre = ans_ex
                ans_ex = remove_space_and_ent(llm.fast_run('RandomSeed(%d)\n%s' % (np.random.randint(1,100),prompt)))
                max_try += 1
                if max_try >= 10:
                    ans_ex = None
            
            if ans_ex:
                answer = ans_ex
            else:
                answer = None
            
            if answer:
                question, choices, groud_truth = formulate_QA(question, answer, nameB1, nameB2)
            else:
                choices, groud_truth = '[ERRORC]', '[ERRORG]'

            question_list.append({
                'qid': qid,
                'question': question,
                'answer': answer,
                'target_step_id': [qid*2, qid*2+1],
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
    output_path = '03_comparative_roles.json'
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


def generate_compare_event_03b(graph_list):
    key_features = ['规模', '持续时间']
    # B1_attrs_num = 5
    question_num = 1
    def generate_single_01a(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []

        event_list = graph['work_events'] + graph['rest_events']
        event_C_ids = np.random.choice(range(len(event_list)), size=2, replace=False)
        event_C1, event_C2 = event_list[event_C_ids[0]], event_list[event_C_ids[1]]

        compare_attrs = np.random.choice(key_features, size=question_num, replace=False)
        for compare_attr_k in compare_attrs:
            vC1, vC2 = event_C1[compare_attr_k], event_C2[compare_attr_k]
            nC1, nC2 = event_C1['事件名称'], event_C2['事件名称']

            text = rewrite_message_event("我将要参加的%s，它的%s是%s。" % (nC1, compare_attr_k, vC1), charact)
            message_list.append({
                'name': nC1,
                'attr': (compare_attr_k,vC1),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
            time_clock.update_time()

            text = rewrite_message_event("我将要参加的%s，它的%s是%s。" % (nC2, compare_attr_k, vC2), charact)
            message_list.append({
                'name': nC2,
                'attr': (compare_attr_k,vC2),
                'message': text,
                'time': time_clock.get_current_time(),
                'place': graph['user_profile']['工作地']
            })
            time_clock.update_time()

        for qid in range(question_num):
            compare_attr_k = compare_attrs[qid]
            vC1, vC2 = event_C1[compare_attr_k], event_C2[compare_attr_k]
            nC1, nC2 = event_C1['事件名称'], event_C2['事件名称']

            for noise_event_id in range(len(event_list)):
                if noise_event_id not in event_C_ids:
                    v = event_list[noise_event_id][compare_attr_k]
                    n = event_list[noise_event_id]['事件名称']

                    text = rewrite_message_event("%s的%s是%s。" % (n, compare_attr_k, v), charact)
                    message_list.append({
                        'name': n,
                        'attr': (compare_attr_k,v),
                        'message': text,
                        'time': time_clock.get_current_time(),
                        'place': graph['user_profile']['工作地']
                    })
                    time_clock.update_time()

            prompt = '[信息] %s的%s是%s；%s的%s是%s。\n' % (nC1,compare_attr_k, vC1, nC2,compare_attr_k,vC2)
            prompt += '基于给出的两个活动的信息，请从 %s 角度，帮我生成一个可以回答的，含有比较的问题。' % compare_attr_k
            prompt += '只输出所生成问题，不要输出其他描述信息。\n'
            prompt += '输出示例：创新大赛和美食节哪个活动的规模更大？'
            question = remove_space_and_ent(llm.fast_run(prompt))
        
            prompt = '[信息] %s的%s是%s；%s的%s是%s。\n' % (nC1,compare_attr_k, vC1, nC2,compare_attr_k,vC2)
            prompt += '基于给出的两个活动的信息，请帮我回答问题：%s。\n' % question
            prompt += '如果无法根据信息回答问题，请输出 无法判断；如果两者一样，请输出 两者一样；否则，答案应该是 %s 或 %s。\n' % (nC1, nC2)

            prompt += '只输出该问题的答案，不要输出其他描述信息，不要输出推理或解释。\n'
            prompt += '输出示例：无法判断'

            ans_pre = remove_space_and_ent(llm.fast_run(prompt))
            ans_ex = remove_space_and_ent(llm.fast_run('RandomSeed(%d)\n%s' % (np.random.randint(1,100),prompt)))
            max_try = 0
            while ans_ex != ans_pre:
                ans_pre = ans_ex
                ans_ex = remove_space_and_ent(llm.fast_run('RandomSeed(%d)\n%s' % (np.random.randint(1,100),prompt)))
                max_try += 1
                if max_try >= 10:
                    ans_ex = None
            
            if ans_ex:
                answer = ans_ex
            else:
                answer = None
            
            if answer:
                question, choices, groud_truth = formulate_QA(question, answer, nC1, nC2)
            else:
                choices, groud_truth = '[ERRORC]', '[ERRORG]'
            
            question_list.append({
                'qid': qid,
                'question': question,
                'answer': answer,
                'target_step_id': [qid*2, qid*2+1],
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
    output_path = '03_comparative_events.json'
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

    generate_compare_role_03a(graph_list[:50])
    generate_compare_event_03b(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions()