import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
import string

from common import rewrite_message, rewrite_question, llm, get_choices, formulate_QA

trajectory_per_graph = 1

all_item_dict = {
    '手机': ['苹果iPhone 13', '苹果iPhone 14', '华为Mate 60', '华为P50', '小米11', '小米12', 'OPPO Reno7', 'OPPO Find X3'],
    '家电': ['小米电视4A', '格力空调', '海尔冰箱', '美的电饭煲', '西门子洗衣机', '松下空气净化器', '索尼电视', '博世洗碗机', 'LG双门冰箱', '戴森吸尘器'],
    '美妆': ['兰蔻奇迹香水', '雅诗兰黛小棕瓶', '魅可子弹头唇膏', '迪奥真我香水', '魅可矿质柔润遮瑕膏', '倩碧特效润肤露', '资生堂红妍肌活精华液', 'SK-II护肤精华露', '雅漾舒护活泉水喷雾', '欧莱雅奇焕润发精油'],
    '运动鞋': ['耐克Air Zoom Pegasus', '阿迪达斯UltraBoost', '锐步Floatride Run Fast', '新百伦Fresh Foam 1080v10', '亚瑟士Gel-Kayano 26', '安德玛Curry 7', '彪马Jam Spark', '斯凯奇Go Walk 5', '乔丹Air Jordan XXXV', '匡威Chuck Taylor All Star']
}

all_place_dict = {
    '小区': ['阳光小区', '绿洲家园', '世纪花园', '锦绣江南', '幸福家园', '金色家园', '龙湖小区', '紫荆花园', '万科城', '保利社区'],
    '公园': ['深圳湾公园', '莲花山公园', '东湖公园', '笔架山公园', '洪湖公园', '中心公园', '荔枝公园', '人民公园', '儿童公园', '园博园'],
    '商场': ['万达广场', '大悦城', '恒隆广场', '银泰百货', '万象城', '印象城', '大润发'],
    '医院': ['人民医院', '中医院', '妇幼保健院', '肿瘤医院', '口腔医院', '眼科医院', '胸科医院', '骨科医院', '皮肤病医院', '精神卫生中心'],
    '景区': ['欢乐谷', '世界之窗', '民俗文化村', '华侨城', '海洋世界', '野生动物园', '欢乐海岸', '海滨公园']
}


def generate_posthoc_of_addition(graph_list):
    B1_attrs_num = 5
    question_num = 1
    def generate_condition_facts_05_item(graph):
        user = graph['user_profile']
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

        question = "对于我%s的%s，以下哪个是最可能的描述？" % (item['关系'],item['物品类型'])
        answer = item['物品评价']
        
        other_answers = []
        other_item_types = list(set(['手机', '家电', '美妆', '运动鞋']) - set([item['物品类型']]))

        for oit in other_item_types:
            ot_item = np.random.choice(all_item_dict[oit], size=1, replace=False)[0]
            prompt = '请你扮演一个%s的%s，职业是%s，在%s担任%s，兴趣爱好是%s，性格是%s。' % (user['年龄'],user['性别'],user['职业'],user['单位名称'],user['职位'],user['兴趣爱好'],user['性格'])
            prompt += '请你扮演该角色，对你%s的%s生成一个评价。只输出评价，不要输出其他的任何信息。\n' % (oit, ot_item)
            prompt += '输出示例：我感觉苹果iPhone 13用起来不错，尤其是软件生态很好，而且拍照也不错，不足之处是续航有点差。'

            other_answers.append(remove_space_and_ent(llm.fast_run(prompt)))
        
        question, choices, groud_truth = formulate_QA(question, answer, other_answers= other_answers)

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

    def generate_condition_facts_05_place(graph):
        user = graph['user_profile']
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

        question = "对于我%s的%s，以下哪个是最可能的描述？" % (place['关系'],place['地点类型'])
        answer = place['地点评价']

        other_answers = []
        other_place_types = np.random.choice(list(set(['小区', '公园', '商场', '医院', '景区']) - set([place['地点类型']])),size=3,replace=False).tolist()
        for opt in other_place_types:
            ot_place = np.random.choice(all_place_dict[opt], size=1, replace=False)[0]
            prompt = '请你扮演一个%s的%s，职业是%s，在%s担任%s，兴趣爱好是%s，性格是%s。' % (user['年龄'],user['性别'],user['职业'],user['单位名称'],user['职位'],user['兴趣爱好'],user['性格'])
            prompt += '请你扮演该角色，对你%s的%s生成一个评价。只输出评价，不要输出其他的任何信息。\n' % (opt, ot_place)
            prompt += '输出示例：我感觉苹果iPhone 13用起来不错，尤其是软件生态很好，而且拍照也不错，不足之处是续航有点差。'

            other_answers.append(remove_space_and_ent(llm.fast_run(prompt)))
        
        question, choices, groud_truth = formulate_QA(question, answer, other_answers= other_answers)

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

    output_path_item = '05_post_processing_items.json'
    output_path_place = '05_post_processing_places.json'
    data_list_item = []
    data_list_place = []
    for index, graph in enumerate(graph_list):
        print('--- %d Graph ---' % index)
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_condition_facts_05_item(graph)
            data_list_item.append({
                'tid': len(data_list_item),
                'message_list': message_list,
                'question_list': question_list
            })
            print(len(data_list_item)-1, 'Finish!')
        for trj in range(trajectory_per_graph):
            message_list, question_list = generate_condition_facts_05_place(graph)
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
        generate_posthoc_of_addition(graph_list)
        
    else:
        generate_posthoc_of_addition(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions(demo_mode=True)