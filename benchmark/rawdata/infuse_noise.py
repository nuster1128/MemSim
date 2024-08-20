import json
import numpy as np
import random

np.random.seed(1128)
random.seed(1128)

def infuse_single_trajectory(traj, noise_pool, OT):
    if OT == 0:
        noisy_traj = {}
        noisy_traj['tid'] = traj['tid']
        noisy_traj['message_list'] = ['%s (消息地点 %s, 消息时间 %s)' % (m['message'], m['place'], m['time']) for m in traj['message_list']]
        noisy_traj['QA'] = traj['QA']
        return noisy_traj

    noisy_traj = {}
    noisy_traj['tid'] = traj['tid']
    new_message_list = []
    raw_message_list = traj['message_list']

    total_step = (1 + OT) * len(raw_message_list)

    tmp = sorted(np.random.choice(range(total_step), size=len(raw_message_list), replace=False))
    relocate_dict = {int(tmp):index for index, tmp in enumerate(tmp)}
    reverse_relocate_dict = {index:int(tmp) for index, tmp in enumerate(tmp)}
    
    noise_list = np.random.choice(noise_pool, size=OT*len(raw_message_list), replace=False)
    noise_id = 0
    for index in range(total_step):
        if index in relocate_dict:
            re_index = relocate_dict[index]
            new_message_list.append('%s (消息地点 %s, 消息时间 %s)' % (raw_message_list[re_index]['message'], raw_message_list[re_index]['place'], raw_message_list[re_index]['time']))
        else:
            new_message_list.append(noise_list[noise_id])
            noise_id += 1
    
    noisy_traj['message_list'] = new_message_list
    noisy_traj['QA'] = traj['QA']
    noisy_traj['QA']['target_step_id'] = [reverse_relocate_dict[step_id] for step_id in noisy_traj['QA']['target_step_id']]

    return noisy_traj


def noise_infuse(OT, dir_path):
    noise_path = 'noise_pool.json'
    pure_path = 'memdaily.json'
    output_path = '%s/memdaily_%d.json' % (dir_path, OT)

    output_data = {}

    with open(noise_path,'r', encoding='utf-8') as f:
        noise_pool = json.load(f)

    with open(pure_path,'r', encoding='utf-8') as f:
        pure_data = json.load(f)


    for QAtype, QAtype_data in pure_data.items():
        output_data[QAtype] = {}
        for scenario, scenario_data in QAtype_data.items():
            output_data[QAtype][scenario] = []
            for traj in scenario_data:
                noisy_traj = infuse_single_trajectory(traj, noise_pool, OT)
                output_data[QAtype][scenario].append(noisy_traj)
            print(OT,QAtype, scenario, 'Finish!')
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(output_data,f, indent=4,ensure_ascii=False)


if __name__ == '__main__':
    noise_infuse(0, 'memdaily_with_noise') # 1*
    noise_infuse(9, 'memdaily_with_noise') # 10*
    noise_infuse(49, 'memdaily_with_noise') # 50*
    noise_infuse(99, 'memdaily_with_noise') # 100*
    noise_infuse(199, 'memdaily_with_noise') # 200*