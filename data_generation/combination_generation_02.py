import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
import string

from common import rewrite_message, rewrite_question, formulate_QA, llm, rewrite_message_event

trajectory_per_graph = 4 # 10


def get_role_data(graph, time_clock):
    """
    Obtain the message and QA from role entities.
    """
    key_features = ['姓名', '年龄', '身高', '生日', '家乡', '工作地', '教育背景', '职业', '职位', '单位名称', '兴趣爱好', '联系电话', '邮箱地址']
    B1_attrs_num = 5
    question_num = 1

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

        question, choices, groud_truth = formulate_QA(question, answer)
        question_list.append({
            'qid': qid,
            'question': rewrite_question(question),
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

def get_event_data(graph, time_clock):
    key_features = ['主要内容', '地点', '规模', '持续时间']
    B1_attrs_num = 3
    question_num = 1

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

        ### TEST
        # abs_pre = time_clock.get_current_timestamp()
        # [time_clock.update_time() for i in range(30)]
        # rel_pre = '下周五上午九点'
        # rel_cur = time_clock.refine_rel_time(abs_pre, rel_pre, time_clock.get_current_timestamp())
        # raise
        ### TEST END

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

def get_item_data(graph, time_clock):
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

    question = "我%s的%s是什么？" % (item['关系'],item['物品类型'])

    answer = item['物品名称']

    question, choices, groud_truth = formulate_QA(question, answer)

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

def get_place_data(graph, time_clock):
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

    question = "我%s的%s是什么？" % (place['关系'],place['地点类型'])

    answer = place['地点名称']

    question, choices, groud_truth = formulate_QA(question, answer)

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

def get_single_type_data(graph, time_clock, type):
    if type == 'role':
        return get_role_data(graph, time_clock)
    elif type == 'event':
        return get_event_data(graph, time_clock)
    elif type == 'item':
        return get_item_data(graph, time_clock)
    else:
        return get_place_data(graph, time_clock)

def merge_message_and_QA(meta_message_list, meta_question_list, message_list, question_list):
    mid_prefix = len(meta_message_list)
    meta_message_list += [{
        'mid': mid_prefix + mid,
        'message': m['message'],
        'time': m['time'],
        'place': m['place']
    } for mid, m in enumerate(message_list)]

    qid_prefix = len(meta_question_list)
    meta_question_list += [{
        'qid': qid_prefix + q['qid'],
        'question': q['question'],
        'answer': q['answer'],
        'target_step_id': [mid_prefix + ref_id for ref_id in q['target_step_id']],
        'choices': q['choices'],
        'ground_truth': q['ground_truth'],
        'time': q['time']
    } for qid, q in enumerate(question_list)]


def get_new_question_list(meta_question_list):
    def get_choices(ans,other_ans):
        choices = {}

        cvt = {0:'A',1:'B',2:'C',3:'D'}
        ans_tag = np.random.choice(range(4),size=1,replace=False)[0]
        ans_temp = [0 for i in range(4)]
        ans_temp[ans_tag] = 1
        groud_truth = cvt[ans_tag]

        choices[groud_truth] = ans
        for i in range(3):
            for index, t in enumerate(ans_temp):
                if t == 0:
                    ans_temp[index] = 1
                    cur_tag = index
                    break
            choices[cvt[cur_tag]] = other_ans[i]
        choices = {k: choices[k] for k in sorted(choices)}
        return groud_truth,choices
    question_text = ''
    answer_text = ''
    confuse_choices_text_list = ['', '', '']
    target_step_id_list = []
    for qid, q in enumerate(meta_question_list):
        target_step_id_list += q['target_step_id']
        if qid >= 1:
            question_text += '另外，%s' % q['question']
            answer_text += '; %s' % q['answer']
            confuse_choices = [v for k,v in q['choices'].items() if k != q['ground_truth']]
            confuse_choices_text_list = [choice_text+'; %s' % confuse_choices[cid] for cid, choice_text in enumerate(confuse_choices_text_list)]
        else:
            question_text += '%s' % q['question']
            answer_text += '%s' % q['answer']
            confuse_choices = [v for k,v in q['choices'].items() if k != q['ground_truth']]
            confuse_choices_text_list = confuse_choices

    groud_truth, choices = get_choices(answer_text, confuse_choices_text_list)
    return {
        'qid': 0,
        'question': question_text,
        'answer': answer_text,
        'target_step_id': target_step_id_list,
        'choices': choices,
        'ground_truth': groud_truth,
        'time': meta_question_list[-1]['time']
    }
    
def check_both(tp1, tp2):
    if tp1 == 'item' and tp2 == 'place':
        return False
    if tp1 == 'place' and tp2 == 'item':
        return False
    return True

def generate_simple_facts_addition(graph_list):
    B1_attrs_num = 5
    question_num = 1
    def generate_single_02_combination(graph):
        time_clock = TimeClock()
        combination_cand = ['role', 'event', 'item', 'place']
        p = [0.35, 0.35, 0.15, 0.15]
        combination_types = np.random.choice(combination_cand,size=2,replace=False,p = p)

        while combination_types[0] == 'event' or not check_both(combination_types[0], combination_types[1]):
            combination_types = np.random.choice(combination_cand,size=2,replace=False,p = p)

        meta_message_list, meta_question_list = [], []

        for ct in combination_types:
            message_list, question_list = get_single_type_data(graph, time_clock, ct)
            merge_message_and_QA(meta_message_list, meta_question_list, message_list, question_list)
        # print(meta_message_list,meta_question_list)

        meta_question_list = [get_new_question_list(meta_question_list)]
        
        return meta_message_list, meta_question_list
        
    data_list = []
    output_path = '02_conditional_hybrid.json'
    for index, graph in enumerate(graph_list):
        print('--- %d Graph ---' % index)
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_single_02_combination(graph)
            data_list.append({
                'tid': len(data_list),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list)-1, 'Finish!')
    
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4,ensure_ascii=False)

def generate_memory_and_questions(demo_mode = False):
    profiles_path = 'graphs.json'
    with open(profiles_path,'r', encoding='utf-8') as f:
        graph_list = json.load(f)

    if not demo_mode:
        generate_simple_facts_addition(graph_list)
    else:
        generate_simple_facts_addition(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions(demo_mode=True)