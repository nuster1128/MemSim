import json
import numpy as np
from utils import create_LLM, remove_space_and_ent, TimeClock
from common import rewrite_message, rewrite_question, llm, get_choices, formulate_QA, rewrite_message_event
import string

trajectory_per_graph = 2

def generate_other_choices_05(attr, answer, fea):
    def get_other_answers(res):
        ans_list = []
        lines = [l for l in res.splitlines() if l != '']
        if len(lines) != 3:
            return None
        for line in lines:
            if line[:len('A. ')] in ['A. ', 'B. ', 'C. ']:
                ans_list.append(line[len('A. '):])
            else:
                return None
        return ans_list

    prompt = '问题： 请问它的%s是什么？\n' % attr
    prompt += '正确答案: %s\n' % answer
    prompt += '请帮我根据上述问题和答案，生成三个不同的混淆选项。\n'
    prompt += '每一个混淆选项需要和答案的长度相似，且内容区分明显。\n'
    prompt += '输出应该遵循以下示例格式：\n'
    prompt += 'A. 成都\n'
    prompt += 'B. 北京\n'
    prompt += 'C. 上海\n'

    res = llm.fast_run(prompt)
    other_answers = get_other_answers(res)

    max_tries = 10
    while not other_answers:
        res = llm.fast_run(prompt)
        max_tries -= 1
        other_answers = get_other_answers(res)
        if max_tries <= 0:
            other_answers = ['不知道', '都不正确', '未提及答案']

    other_feas = []

    for oa in other_answers:
        prompt = '请你用一句简短的话，给出 %s 最独特的特点，但不要包含%s。\n' % (oa,oa)
        prompt += '只需要输出特点，不需要输出其他描述性内容。\n'
        prompt += '示例输出：%s' % fea

        other_feas.append(remove_space_and_ent(llm.fast_run(prompt)))
    return other_feas


def generate_posthoc_role_05a(graph_list):
    key_features = ['姓名', '年龄', '生日', '家乡', '工作地', '教育背景', '职业', '职位', '兴趣爱好', '联系电话', '邮箱地址']
    targeted_features = ['姓名', '生日', '工作地', '职业', '兴趣爱好', '联系电话', '邮箱地址']
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
        attrs_other = np.random.choice(key_features, size=B1_attrs_num-1, replace=False)
        attrs_target = np.random.choice(list(set(targeted_features)-set(attrs_other)), size=1, replace=False)

        for k in attrs_target:
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

        for k in attrs_other:
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
            inx_01, inx_02 = np.random.choice(range(len(message_list[1:])), size=1, replace=False)[0] + 1, 0

            real_attr_01, real_attr_02 = message_list[inx_01], message_list[inx_02]

            other_answers = None
            for noise_role_id in range(len(role_list)):
                if noise_role_id != role_B1_id:
                    name, k = role_list[noise_role_id]['姓名'], real_attr_02['attr'][0]
                    v = role_list[noise_role_id][k]
                    text = rewrite_message("你好，小艺助手，%s的%s是%s。" % (name, k, v), charact)

                    noise_message_list.append({
                        'message': text,
                        'time': time_clock.get_current_time(),
                        'place': graph['user_profile']['工作地']
                    })
                    time_clock.update_time()

            if real_attr_02['attr'][0] in ['联系电话']:
                sum_num = int(np.random.choice(range(2,6+1),size=1,replace=False)[0])
                question = "那个%s是%s的人，其%s的后%d位之和是多少？" % (real_attr_01['attr'][0], real_attr_01['attr'][1], real_attr_02['attr'][0],sum_num)
                answer = str(sum(int(digit) for digit in real_attr_02['attr'][1][-sum_num:]))
            elif real_attr_02['attr'][0] in ['邮箱地址']:
                question = "那个%s是%s的人，其%s的后缀是多少？" % (real_attr_01['attr'][0], real_attr_01['attr'][1], real_attr_02['attr'][0])
                answer = real_attr_02['attr'][1][real_attr_02['attr'][1].find('@'):]
            elif real_attr_02['attr'][0] in ['姓名']:
                question = "那个%s是%s的人，其%s的有几个字？" % (real_attr_01['attr'][0], real_attr_01['attr'][1], real_attr_02['attr'][0])
                answer = '%s个字' % len(real_attr_02['attr'][1])
                other_answers = ['%s个字' % (i + len(real_attr_02['attr'][1])) for i in [-1,1,2]]
            elif real_attr_02['attr'][0] in ['生日']:
                question = "那个%s是%s的人，其生日是在哪个季节？" % (real_attr_01['attr'][0], real_attr_01['attr'][1])
                month_dig = int(real_attr_02['attr'][1].split('月')[0])
                print('Month: %d月(%s)' % (month_dig,real_attr_02['attr'][1]))
                if 3<= month_dig <= 5:
                    answer = '春季'
                elif 6 <= month_dig <= 8:
                    answer = '夏季'
                elif 9 <= month_dig <= 11:
                    answer = '秋季'
                else:
                    answer = '冬季'
                other_answers = [ season for season in ['春季', '夏季', '秋季', '冬季'] if answer != season]
            elif real_attr_02['attr'][0] in ['职业']:
                work_explain = {
                    '医生': '救治病人，保障人民健康',
                    '教师': '教书育人，传道授业解惑',
                    '工程师': '设计、建造和维护工程项目',
                    '律师': '维护法律尊严，为当事人提供法律服务',
                    '警察': '维护社会治安，保障人民生命财产安全',
                    '护士': '协助医生救治病人，提供护理服务',
                    '厨师': '烹制美食，满足顾客味蕾',
                    '程序员': '编写代码，开发软件',
                    '销售人员': '推广产品，实现销售目标',
                    '会计': '管理财务，确保企业经济活动合规',
                    '银行柜员': '办理金融业务，服务客户',
                    '设计师': '创新设计，美化生活',
                    '记者': '报道新闻，传播信息',
                    '快递员': '配送货物，确保快速送达',
                    '建筑工人': '建造房屋，建设基础设施',
                    '翻译员': '沟通语言障碍，促进交流',
                    '农民': '勤恳劳动，种植庄稼和蔬果',
                    '乘务员': '为旅客提供优质服务'
                }

                question = "那个%s是%s的人，其职业的主要职责是什么？" % (real_attr_01['attr'][0], real_attr_01['attr'][1])
                answer = work_explain[real_attr_02['attr'][1]]

                other_works = np.random.choice(list(set(work_explain.keys())-set([real_attr_02['attr'][1]])), size=3, replace=False)
                other_answers = [work_explain[owk] for owk in other_works]
            elif real_attr_02['attr'][0] in ['兴趣爱好']:
                hobby_explain = {
                    '阅读': '享受文字之美，丰富内心世界',
                    '旅游': '读万卷书，不如行万里路',
                    '运动': '增强体质，保持健康',
                    '听音乐': '放松心情，在耳机中感受旋律之美',
                    '看电影': '欣赏影视作品，体验不同人生',
                    '绘画': '用画笔表达情感，创造美',
                    '摄影': '用镜头捕捉瞬间，记录生活',
                    '舞蹈': '翩翩起舞，表达自我，享受节奏',
                    '瑜伽': '放松身心，修身养性，享受瑜伽',
                    '健身': '杠铃，卧推，塑造体形',
                    '美食烹饪': '制作美食，享受烹饪乐趣',
                    '园艺': '培育植物，亲近自然',
                    '手工艺': '动手创造，体验手工艺术',
                    '模型制作': '精细制作，展现创造力，动手制作模型',
                    '集邮': '收集邮票，了解历史文化',
                    '跑步': '有氧运动，提高心肺功能',
                    '自行车骑行': '骑行探索，享受户外乐趣',
                    '游泳': '水中运动，全面锻炼身体',
                    '登山': '挑战自我，征服高峰',
                    '攀岩': '极限运动，锻炼勇气与技巧',
                    '钓鱼': '静心等待，享受垂钓乐趣',
                    '打高尔夫球': '优雅运动，锻炼协调性，梦想一杆进洞',
                    '下棋': '智力游戏，锻炼逻辑思维，棋手无双',
                    '玩电子游戏': '电竞虚拟世界，体验游戏乐趣',
                    '编程': '编写软件，解决问题',
                    '学习外语': '掌握新语言，拓宽视野',
                    '写作': '写作表达思想，记录生活',
                    '书法': '练习书法，传承文化',
                    '戏剧': '观赏戏剧，体验人生百态',
                    '听音乐会': '聆听现场音乐，享受艺术氛围'
                }
                question = "那个%s是%s的人，其兴趣爱好的主要内容是什么？" % (real_attr_01['attr'][0], real_attr_01['attr'][1])
                answer = hobby_explain[real_attr_02['attr'][1]]

                other_hobbies = np.random.choice(list(set(hobby_explain.keys())-set([real_attr_02['attr'][1]])), size=3, replace=False)
                other_answers = [hobby_explain[owk] for owk in other_hobbies] 
            elif real_attr_02['attr'][0] in ['工作地']:
                place_explain = {
                    '北京': '首都，政治文化中心',
                    '上海': '国际都市，经济金融中心',
                    '广东广州': '珠三角省会城市，历史悠久的商贸中心',
                    '广东深圳': '经济特区，科技创新的重要城市',
                    '江苏南京': '历史文化名城，长三角地区的重要省会城市',
                    '浙江杭州': '西湖风景秀丽，互联网产业发达的城市',
                    '山东青岛': '海滨城市，著名的旅游和港口城市，以啤酒著称'
                }

                question = "那个%s是%s的人，以下哪一项符合其工作地的描述？" % (real_attr_01['attr'][0], real_attr_01['attr'][1])
                answer = place_explain[real_attr_02['attr'][1]]

                other_places = np.random.choice(list(set(place_explain.keys())-set([real_attr_02['attr'][1]])), size=3, replace=False)
                other_answers = [place_explain[owk] for owk in other_places] 
            else:
                raise "Role targeted attr error: %s." % real_attr_02['attr'][0]


            question, choices, groud_truth = formulate_QA(question, answer, other_answers=other_answers)
            print(question)
            print(answer)
            print(other_answers)
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
    output_path = '05_post_processing_roles.json'
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

def generate_posthoc_event_05b(graph_list):
    key_features = ['地点', '时间', '主要内容']
    B1_attrs_num = 3
    question_num = 1
    def generate_single_01a(graph):
        time_clock = TimeClock()
        charact = graph['user_profile']['性格']
        message_list = []
        noise_message_list = []
        question_list = []
        second_sample_pool = []

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

            if message_list[-1]['attr'][0] in ['事件类型','地点', '时间']:
                second_sample_pool.append(len(message_list)-1)

        for qid in range(question_num):
            inx_01 = np.random.choice(range(len(message_list)), size=1, replace=False)[0]
            while message_list[inx_01]['attr'][0] in ['事件类型']:
                inx_01 = np.random.choice(range(len(message_list)), size=1, replace=False)[0]
            inx_02 = np.random.choice(second_sample_pool, size=1, replace=False)[0]
            while inx_02 == inx_01:
                inx_02 = np.random.choice(second_sample_pool, size=1, replace=False)[0]
                print('Resample index_02')
            real_attr_01, real_attr_02 = message_list[inx_01], message_list[inx_02]

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
            

            if real_attr_02['attr'][0] in ['地点']:
                city_explain = {
                        '山东济南': '山东省会，泉城',
                        '广东广州': '广东省会，历史悠久的商贸中心',
                        '江苏南京': '江苏省会，历史文化名城',
                        '浙江杭州': '浙江省会，风景秀丽的城市',
                        '福建福州': '福建省会，温泉之城',
                        '河南郑州': '河南省会，交通枢纽',
                        '湖北武汉': '湖北省会，长江中游的重要城市',
                        '湖南长沙': '湖南省会，娱乐之都',
                        '四川成都': '四川省会，天府之国',
                        '重庆': '直辖市，山城',
                        '北京': '首都，政治文化中心',
                        '上海': '直辖市，珠三角国际大都市',
                        '天津': '直辖市，北方重要港口',
                        '河北石家庄': '河北省会，新兴城市',
                        '山西太原': '山西省会，能源基地',
                        '辽宁沈阳': '辽宁省会，东北重工业基地',
                        '吉林长春': '吉林省会，汽车城',
                        '黑龙江哈尔滨': '黑龙江省会，冰城',
                        '江西南昌': '江西省会，英雄城市',
                        '安徽合肥': '安徽省会，科教之城',
                        '江苏苏州': '江南水乡，经济发达，珠三角非省会的重要城市',
                        '浙江宁波': '浙江港口城市，经济发展迅速',
                        '福建厦门': '海岛城市，经济特区，鼓浪屿',
                        '山东青岛': '海滨城市，旅游胜地，以啤酒著称',
                        '广东深圳': '经济特区，科技创新城市，毗邻香港'
                    }
                
                question = "那个%s是%s的活动，哪一个符合它的活动地点描述？" % (real_attr_01['attr'][0], real_attr_01['attr'][1])
                answer = city_explain[real_attr_02['attr'][1]]

                other_cities = np.random.choice(list(set(city_explain.keys())-set([real_attr_02['attr'][1]])), size=3, replace=False)
                other_answers = [city_explain[owk] for owk in other_cities] 

            elif real_attr_02['attr'][0] in ['时间']:
                given_time = real_attr_02['attr'][1]
                pre_abs_time = message_list[inx_02]['time']
                if given_time[0] == '下':   # Given time is relative time.
                    new_given_time = time_clock.reltime_to_abstime(time_clock.format_time_to_timestamp(pre_abs_time), given_time)
                else: # Given time is absolute time.
                    new_given_time = time_clock.calculate_reltime(time_clock.format_time_to_timestamp(pre_abs_time), given_time)
                
                answer = new_given_time
                other_answers = None
                if real_attr_01['attr'][0] == '主要内容':
                    question = "那个%s是\"%s\"的活动，其%s是什么？" % (real_attr_01['attr'][0], real_attr_01['attr'][1], real_attr_02['attr'][0])
                else:
                    question = "那个%s是%s的活动，其%s是什么？" % (real_attr_01['attr'][0], real_attr_01['attr'][1], real_attr_02['attr'][0])
            elif real_attr_02['attr'][0] in ['事件类型']:
                event_type_explain = {
                    '旅行': '读万卷书，行万里路，体验不同的文化',
                    '阅读': '享受书籍带来的知识和乐趣',
                    '看电影': '欣赏影视作品，放松心情',
                    '健身': '进行体育锻炼，保持身体健康',
                    '美食节': '品尝各种美食，享受美食文化',
                    '音乐会': '聆听音乐，感受音乐的魅力',
                    '艺术展览': '欣赏美术作品，提升审美能力',
                    '朋友聚会': '与朋友相聚，增进友谊',
                    '逛街购物': '逛街选购商品，享受购物乐趣',
                    '户外徒步': '在户外徒步，亲近自然',
                    '商业会议': '商业界的会议，讨论商业策略，促进业务发展',
                    '职业培训': '进行职业技能培训',
                    '招聘会': '寻找工作机会，招聘人才',
                    '产品发布会': '发布新产品，展示企业创新',
                    '行业交流会': '行业内交流，分享经验和见解',
                    '公司团建': '团队建设，增强团队凝聚力，促进员工交流',
                    '年终总结大会': '回顾一年工作，总结经验教训',
                    '项目启动会': '启动新项目，明确目标和计划',
                    '学术交流会': '学术交流会，学术界的交流，分享研究成果',
                    '创新研讨会': '创新研讨会，探讨创新思路，激发创意'
                }

                question = "那个%s是%s的活动，哪一个是它的内容描述？" % (real_attr_01['attr'][0], real_attr_01['attr'][1])
                answer = event_type_explain[real_attr_02['attr'][1]]

                other_event_types = np.random.choice(list(set(event_type_explain.keys())-set([real_attr_02['attr'][1]])), size=3, replace=False)
                other_answers = [event_type_explain[owk] for owk in other_event_types] 
            
            else:
                raise "Event targeted attr error: %s." % real_attr_02['attr'][0]

            question, choices, groud_truth = formulate_QA(question, answer, other_answers=other_answers)
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
    output_path = '05_post_processing_events.json'
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

    # generate_posthoc_role_05a(graph_list[:50])
    generate_posthoc_event_05b(graph_list[:50])


if __name__ == '__main__':
    generate_memory_and_questions()