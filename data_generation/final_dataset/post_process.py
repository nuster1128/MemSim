import os, json
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

llm_model_path = '/data/zhangzeyu/local_llms/glm-4-9b-chat'

tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True, low_cpu_mem_usage=True)

NUM_TO_TYPE = {
    '01':'simple',
    '02':'conditional',
    '03':'comparative',
    '04':'aggregative',
    '05':'post_processing',
    '06':'noisy'
}

def get_sta(sub_data):
    num_question = len(sub_data)
    num_message = 0
    token_list = []
    for tra in sub_data:
        num_message += len(tra['message_list'])
        for m in tra['message_list']:
            tk = tokenizer(m['message'])
            token_list.append(len(tk['input_ids']))
    return num_message, num_question, token_list

def filter_subdata(sub_data):
    new_sub_data = []
    for tp in sub_data:
        message_list = tp['message_list']
        qa = tp['question_list'][0]
        if qa['question'] == '[ERRORQ]' or qa['answer'] == '[ERRORA]':
            continue
        new_sub_data.append({
            'tid': len(new_sub_data),
            'message_list': message_list,
            'QA': qa
        })

    return new_sub_data

def merge(dir_path = '.'):
    sta_list = {
        '%02d' % (i+1): {
        'roles': None,
        'events': None,
        'items': None,
        'places': None,
        'hybrid': None
    } for i in range(6) }

    sta_list['03'] = {
        'roles': None,
        'events': None,
        'hybrid': None
    }

    sta_list['04'] = {
        'roles': None,
        'events': None,
        'hybrid': None
    }

    data_list = {
        NUM_TO_TYPE['%02d' % (i+1)]: {
        'roles': None,
        'events': None,
        'items': None,
        'places': None,
        'hybrid': None
    } for i in range(6) }

    data_list[NUM_TO_TYPE['03']] = {
        'roles': None,
        'events': None,
        'hybrid': None
    }

    data_list[NUM_TO_TYPE['04']] = {
        'roles': None,
        'events': None,
        'hybrid': None
    }

    filenames = os.listdir(dir_path)
    for filename in filenames:
        path = '%s/%s' % (dir_path, filename)
        filename_split = filename.split('_')
        QA_type = filename_split[0]
        scenario_type = filename_split[-1].split('.')[0]
        print(QA_type,scenario_type)

        with open(path,'r', encoding='utf-8') as f:
            sub_data = json.load(f)

        filtered_subdata = filter_subdata(sub_data)

        data_list[NUM_TO_TYPE[QA_type]][scenario_type] = filtered_subdata
        sta_list[QA_type][scenario_type] = get_sta(filtered_subdata)
    
    output_path = 'memdaily.json'
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4,ensure_ascii=False)

    static_res_path = 'data_card_raw.csv'
    data_card = [['Index','Trajectories', 'Messages', 'Questions', 'TPM (avg.)', 'TPM (var.)']]

    for k, v in sta_list.items():
        # output_path = '%s_%s.json' % (k,NUM_TO_TYPE[k])
        # with open(output_path,'w', encoding='utf-8') as f:
        #     json.dump(v, f, indent=4,ensure_ascii=False)
        
        sum_message, sum_QA, token_list = 0, 0, []
        for kk,vv in sta_list[k].items():
            # print(vv)
            sum_message += vv[0]
            sum_QA += vv[1]
            token_list+=vv[2]
        data_card.append([k,sum_QA, sum_message, sum_QA, np.mean(token_list), np.std(token_list)])
        print(k,sta_list[k])
        print(sum_message, sum_QA)
    with open(static_res_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_card)


if __name__ == '__main__':
    merge('messages_and_QAs')