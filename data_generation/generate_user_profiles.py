import csv, json
import numpy as np
from utils import create_LLM, remove_space_and_ent
import random
import string
from common import llm

relation_size = 2
colleague_size = 2

def get_meta_profile():
    meta_profile_path = 'meta_profile.csv'
    meta_profile = {}
    with open(meta_profile_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for index, line in enumerate(reader):
            if index == 0:
                continue
            key, value = line[0], line[1]
            meta_profile[key] = eval(value)
    return meta_profile

def get_gender(rel, self_name = None):
    if rel == 'self':
        prompt = '请你判断名字为%s的性别。如果为男性，请输出1；如果是女性，请输出0。仅输出0或1，不要输出其他任何信息。' % self_name
        prompt += '输出示例：1'
        gd = remove_space_and_ent(llm.fast_run(prompt))
        max_try = 0
        while gd not in ['0', '1']:
            gd = remove_space_and_ent(llm.fast_run(prompt))
            max_try += 1
            if max_try >= 10:
                raise 'Gender Judgement Error!'

        if gd == '1':
            return '男性'
        elif gd == '0':
            return '女性'
        else:
            raise 'Gender Judgement Error!'
    elif rel in ['父亲', '表哥', '表弟', '叔叔', '舅舅']:
        return '男性'
    elif rel in rel in ['母亲', '表姐', '表妹', '婶婶', '舅妈']:
        return '女性'
    else:
        return np.random.choice(['男性', '女性'],size=1,replace=False)[0]

def get_name(self_name, rel, name_str,gender):
    if rel == 'self':
        return self_name
    elif rel in ['父亲', '叔叔']:
        prompt = '请你以 %s 为姓氏，生成一个%s中国人名，不要与 %s 重复。仅输出一个人名，不要输出其他任何信息。' % (self_name[0], gender, name_str)
        prompt += '输出示例：%s' % self_name
        return remove_space_and_ent(llm.fast_run(prompt))
    else:
        prompt = '请你生成一个%s中国人名，不要与 %s 重复。仅输出一个人名，不要输出其他任何信息。' % (gender, name_str)
        prompt += '输出示例：%s' % self_name
        return remove_space_and_ent(llm.fast_run(prompt))
        
def get_age(rel, user):
    if rel == 'self':
        return int(np.random.choice(range(24,42+1),size=1,replace=False)[0])
    elif rel in ['父亲', '母亲', '叔叔', '婶婶', '舅舅', '舅妈']:
        return int(user['年龄'] + np.random.choice(range(20, 25),size=1,replace=False)[0])
    elif rel in ['表哥', '表姐', '上司']:
        return int(user['年龄'] + np.random.choice(range(3, 10),size=1,replace=False)[0])
    elif rel == '同事':
        return int(user['年龄'] + np.random.choice(range(-2, 2),size=1,replace=False)[0])
    else:
        return int(user['年龄'] - np.random.choice(range(2, 4),size=1,replace=False)[0])

def get_height(gender):
    if gender == '男性':
        return int(np.clip(np.random.normal(170,8),150,190))
    else:
        return int(np.clip(np.random.normal(164,8),150,190))

def get_birthday():
    month = random.randint(1, 12)
    if month == 2:
        day = random.randint(1, 28)
    else:
        day_ranges = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day = random.randint(1, day_ranges[month - 1])
    return '%02d月%02d日' % (month, day)

def get_hometown(rel, user, meta_profile):
    if rel == 'self':
        return np.random.choice(meta_profile['家乡'], size=1, replace=False)[0]
    elif rel in ['父亲', '母亲', '表哥', '表姐', '表弟', '表妹', '叔叔', '婶婶', '舅舅', '舅妈']:
        return user['家乡']
    else:
        return np.random.choice(meta_profile['家乡'], size=1, replace=False)[0]

def get_workplace(rel, user, meta_profile):
    if rel == 'self':
        return np.random.choice(meta_profile['工作地'], size=1, replace=False)[0]
    elif rel in ['上司', '下属', '同事']:
        return user['工作地']
    else:
        return np.random.choice(meta_profile['工作地'], size=1, replace=False)[0]

def get_education(rel, user, meta_profile):
    if rel in ['上司', '下属', '同事']:
        return user['教育背景']
    else:
        return np.random.choice(meta_profile['教育背景'],size=1,replace=False)[0]

def get_occupation(rel, user, eduction):
    if rel in ['上司', '下属', '同事']:
        return user['职业']

    choice_list = None
    if eduction == '高中':
        choice_list = ['厨师', '销售人员', '银行柜员', '快递员', '建筑工人', '农民']
    elif eduction == '专科':
        choice_list = ['警察', '护士', '厨师', '销售人员', '银行柜员', '快递员', '建筑工人', '乘务员']
    elif eduction == '本科':
        choice_list = ['医生', '教师', '工程师', '律师', '警察', '护士', '厨师', '程序员', '销售人员', '会计', '银行柜员', '设计师', '记者', '快递员', '建筑工人', '翻译员', '乘务员']
    elif eduction == '硕士':
        choice_list = ['医生', '教师', '工程师', '律师', '护士', '程序员', '会计', '设计师', '记者', '翻译员']
    elif eduction == '博士':
        choice_list = ['医生', '教师', '工程师', '程序员','设计师']
    else:
        raise
    return np.random.choice(choice_list,size=1,replace=False)[0]

def get_position(rel, age, education, occupation, user):
    if rel in ['上司', '下属', '同事']:
        prompt = '张三的职业是%s，他的职位是%s，你是他的%s。' % (user['职业'], user['职位'], rel)
        prompt += '你今年%d岁，教育背景是%s，请随机生成一个你可能的职位，不多于8个字。输出只包括你的职位，不要输出其他任何信息。' % (age, education)
        prompt += '输出示例：经理'
        return remove_space_and_ent(llm.fast_run(prompt))
    else:
        prompt = '你今年%d岁，教育背景是%s，职业是%s，请随机生成一个你可能的职位，不多于8个字。输出只包括你的职位，不要输出其他任何信息。' % (age, education, occupation)
        prompt += '输出示例：员工'
        return remove_space_and_ent(llm.fast_run(prompt))

def get_corp_name(rel, occupation, workplace, user):
    if rel in ['上司', '下属', '同事']:
        return user['单位名称']
    else:
        prompt = '张三的职业是%s，他的工作地点在%s，请你随机虚构一个他可能工作的单位名字，不超过10个字。请只输出单位名字，不要输出其他任何的信息。' % (occupation, workplace)
        prompt += '输出示例：深圳大七才华有限公司'
        return remove_space_and_ent(llm.fast_run(prompt))

def get_hobby(meta_profile):
    return np.random.choice(meta_profile['兴趣爱好'], size=1, replace=False)[0]

def get_charact(meta_profile):
    return np.random.choice(meta_profile['性格'], size=1, replace=False)[0]

def get_phone_number(meta_profile):
    return np.random.choice(meta_profile['联系电话'], size=1, replace=False)[0] + '%08d' % random.randint(0,99999999)

def get_email(name, corp_name, occupation, position, birthday):
    prompt = '你的名字是%s，在%s工作，职业是%s，职位是%s，生日是%s。' % (name, corp_name, occupation,position, birthday)
    prompt += '请随机虚构一个你的电子邮箱地址，必须符合电子邮箱地址的格式。只输出你的电子邮箱地址，不要输出其他的任何信息。'
    prompt += '输出示例：zhangwei1216@szbank.com'
    return remove_space_and_ent(llm.fast_run(prompt))

def get_id_number(age, birthday, gender):
    def validate_id_card(id_number):
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        sum = 0
        for i in range(17):
            sum += int(id_number[i]) * weights[i]
        mod = sum % 11
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
        check_code = check_codes[mod]

        return check_code
    loc_num= np.random.randint(110000,650000)
    birth_num = (2024-age) * 10000 + int(birthday[:2])*100 + int(birthday[3:5])
    order_num = np.random.randint(1,500)*2
    if gender == '男性':
        order_num += 1
    id_num = '%06d%08d%03d' % (loc_num,birth_num,order_num)
    id_num+=validate_id_card(id_num)

    return id_num

def get_passport_number():
    return 'N' + string.ascii_uppercase[np.random.choice(range(0,26), size=1, replace=False)[0]] + '%07d' % random.randint(0,9999999)

def get_bank_number(meta_profile):
    return np.random.choice(meta_profile['银行卡号码'], size=1, replace=False)[0] + '%05d%05d' % (random.randint(0,99999), random.randint(0,99999))

def get_driver_number(prefix, age, birthday, gender):
    def validate_id_card(id_number):
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        sum = 0
        for i in range(17):
            sum += int(id_number[i]) * weights[i]
        mod = sum % 11
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
        check_code = check_codes[mod]

        return check_code
    loc_num= prefix
    birth_num = (2024-age) * 10000 + int(birthday[:2])*100 + int(birthday[3:5])
    order_num = np.random.randint(1,500)*2
    if gender == '男性':
        order_num += 1
    id_num = '%s%08d%03d' % (loc_num,birth_num,order_num)
    id_num+=validate_id_card(id_num)

    return id_num

def generate_role_profile(g_index, meta_profile, rel, user, name_str):
    role_info = {}

    gender = get_gender(rel,meta_profile['姓名'][g_index])
    role_info['性别'] = gender

    name = get_name(meta_profile['姓名'][g_index], rel, name_str, gender)
    role_info['关系'] = rel
    role_info['姓名'] = name

    role_info['年龄'] = get_age(rel, user)
    role_info['身高'] = get_height(gender)
    role_info['生日'] = get_birthday()
    role_info['家乡'] = get_hometown(rel, user, meta_profile)
    role_info['工作地'] = get_workplace(rel, user, meta_profile)
    
    role_info['教育背景'] = get_education(rel, user, meta_profile)
    role_info['职业'] = get_occupation(rel, user, role_info['教育背景'])
    role_info['职位'] = get_position(rel, role_info['年龄'],role_info['教育背景'],role_info['职业'], user)
    role_info['单位名称'] = get_corp_name(rel, role_info['职业'], role_info['工作地'], user)
    role_info['兴趣爱好'] = get_hobby(meta_profile)
    role_info['性格'] = get_charact(meta_profile)

    role_info['联系电话'] = get_phone_number(meta_profile)
    role_info['邮箱地址'] = get_email(name, role_info['单位名称'], role_info['职业'], role_info['职位'], role_info['生日'])
    if rel == 'self':
        role_info['身份证号码'] = get_id_number(role_info['年龄'], role_info['生日'], gender)
        role_info['护照号码'] = get_passport_number()
        role_info['银行卡号码'] = get_bank_number(meta_profile)
        role_info['驾驶证号码'] = get_driver_number(role_info['身份证号码'][:6],role_info['年龄'], role_info['生日'], gender)

    return role_info

def get_event_content(t, user):
    if t in ['旅行', '阅读', '看电影', '健身', '美食节', '音乐会', '艺术展览', '朋友聚会', '逛街购物', '户外徒步']:
        hobby = user['兴趣爱好']
        prompt = '你的兴趣爱好是%s，你现在要去参加一个%s活动，请你结合自己的爱好，随机生成一个该活动内容的概述，40字以内。' % (hobby, t)
        prompt += '输出示例：展示各国美食，提供烹饪示范、食品品尝和采购，促进文化交流。'
        return remove_space_and_ent(llm.fast_run(prompt))
    else:
        prompt = '你的职业是%s，职位是%s，工作地是%s，你现在要去参加一个%s活动，请你根据活动类型和自身情况，随机生成一个该活动内容的概述，40字以内。' % (user['职业'], user['职位'], user['工作地'], t)
        prompt += '输出示例：展开人工智能领域的学术讨论，探讨AI技术进展、挑战与应用，汇聚全球专家交流与合作。'
        return remove_space_and_ent(llm.fast_run(prompt))

def get_event_place(meta_profile, user):
    place_type = np.random.choice(['家乡', '工作地', '大城市'], size=1, replace=False, p=[0.1,0.8,0.1])[0]
    if place_type == '大城市':
        return np.random.choice(meta_profile['工作地'], size=1, replace=False)[0]
    else:
        return user[place_type]

def get_event_time():
    time_type = np.random.choice(['相对时间', '绝对时间'], size=1, replace=False, p=[0.5, 0.5])[0]

    if time_type == '相对时间':
        prefix = np.random.choice(['下周', '下下周'], size=1, replace=False)[0]
        mid = np.random.choice(['一', '二', '三', '四', '五', '六', '日'], size=1, replace=False)[0]
        ex = np.random.choice(['上午九点', '下午两点', '晚上七点'], size=1, replace=False)[0]
        return prefix + mid + ex
    else:
        date = random.randint(1,7)
        clk = np.random.choice(['9:00', '14:00', '19:00'])
        return '2024-04-1%d %s' % (date, clk)

def get_event_name(type, content, place):
    prompt = '现在有一个%s活动，主要内容是%s。' % (type, content)
    prompt += '请你帮我为这个活动取个名字，要求10个字以内，不要出现地点。输出只包括活动的名字，不要输出其他任何信息。'
    prompt += '输出示例：有机蔬菜博览会'
    return remove_space_and_ent(llm.fast_run(prompt))

def get_event_scale(meta_profile, content, name):
    prompt = '现在有一个%s活动，活动名称是“%s”，它的主要内容是%s。\n' % (type, name, content)
    prompt += '请你帮预估一下这个活动的规模，从“十人”、“百人”和“千人”中选取一个最合适的。\n'
    prompt += '输出只包括预估的规模，不要输出其他任何信息。\n'
    prompt += '输出示例：百人'

    res = remove_space_and_ent(llm.fast_run(prompt))

    max_try = 0
    if res not in ['十人', '百人', '千人']:
        max_try += 1
        if max_try >= 10:
            print(res)
            raise "Excess Max Tries."
        res = remove_space_and_ent(llm.fast_run(prompt))

    scale = ['一','二','三','四','五','六','七','八','九']
    target_scale = np.random.choice(scale, size=1, replace=False)[0]

    return '%s%s' % (target_scale,res)

def get_event_lasting(type, content, name):
    prompt = '现在有一个%s活动，活动名称是“%s”，它的主要内容是%s。' % (type, name, content)
    prompt += '请你帮预估一下这个活动的持续时间，从“天”、“周”和“月”中选取一个最合适的，表示几天、几周和几个月。\n'
    prompt += '输出只包括预估的规模，不要输出其他任何信息。\n'
    prompt += '输出示例：天'

    res = remove_space_and_ent(llm.fast_run(prompt))

    max_try = 0
    if res not in ['天', '周', '月']:
        max_try += 1
        if max_try >= 10:
            print(res)
            raise "Excess Max Tries."
        res = remove_space_and_ent(llm.fast_run(prompt))

    if res != '天':
        res = '个%s' % res

    scale = ['一','两','三','四','五','六','七','八','九']
    target_scale = np.random.choice(scale, size=1, replace=False)[0]

    return '%s%s' % (target_scale,res)

def generate_event_profile(meta_profile, t, user):
    event_info = {}

    event_info['事件类型'] = t
    event_info['主要内容'] = get_event_content(t, user)
    event_info['地点'] = get_event_place(meta_profile, user)
    event_info['时间'] = get_event_time()
    event_info['事件名称'] = get_event_name(t, event_info['主要内容'], event_info['地点'])
    event_info['规模'] = get_event_scale(t, event_info['主要内容'], event_info['事件名称'])
    event_info['持续时间'] = get_event_lasting(t, event_info['主要内容'], event_info['事件名称'])

    return event_info

def get_item_type(user):
    if user['性别'] == '男性':
        cand =  ['手机', '家电', '运动鞋']
    else:
        cand = ['手机', '家电', '美妆']
    
    return np.random.choice(cand,size=1,replace=False)[0]

def get_item_name(meta_profile, item_type):
    return np.random.choice(meta_profile[item_type], size=1,replace=False)[0]

def get_item_comment(t, item_name, user):
    prompt = '请你扮演一个%s的%s，职业是%s，在%s担任%s，兴趣爱好是%s，性格是%s。' % (user['年龄'],user['性别'],user['职业'],user['单位名称'],user['职位'],user['兴趣爱好'],user['性格'])
    prompt += '请你扮演该角色，对你%s的%s生成一个评价。只输出评价，不要输出其他的任何信息。\n' % (t, item_name)
    prompt += '输出示例：我感觉苹果iPhone 13用起来不错，尤其是软件生态很好，而且拍照也不错，不足之处是续航有点差。'
    
    return remove_space_and_ent(llm.fast_run(prompt))

def generate_item_profile(meta_profile, t, user):
    item_info = {}

    item_info['关系'] = t
    item_info['物品类型'] = get_item_type(user)
    item_info['物品名称'] = get_item_name(meta_profile,item_info['物品类型'])
    item_info['物品评价'] = get_item_comment(t, item_info['物品名称'], user)
    return item_info

def get_place_type(t):
    if t == '居住':
        return '小区'
    else:
        return np.random.choice(['公园', '商场', '医院', '景区'],size=1,replace=False)[0]

def get_place_name(meta_profile, place_type):
    return np.random.choice(meta_profile[place_type], size=1,replace=False)[0]

def get_place_comment(t, item_name, user):
    prompt = '请你扮演一个%s的%s，职业是%s，在%s担任%s，兴趣爱好是%s，性格是%s。' % (user['年龄'],user['性别'],user['职业'],user['单位名称'],user['职位'],user['兴趣爱好'],user['性格'])
    prompt += '请你扮演该角色，对你%s的%s生成一个评价。只输出评价，不要输出其他的任何信息。\n' % (t, item_name)
    prompt += '输出示例：我居住的东方小区很不错，交通便利，旁边就是地铁站，而且离单位很近，但就是周围停车不太方便。'
    
    return remove_space_and_ent(llm.fast_run(prompt))

def generate_place_profile(meta_profile, t, user):
    place_info = {}

    place_info['关系'] = t
    place_info['地点类型'] = get_place_type(t)
    place_info['地点名称'] = get_place_name(meta_profile, place_info['地点类型'])
    place_info['地点评价'] = get_place_comment(t, place_info['地点名称'], user)

    return place_info

def generate_single_graph(meta_profile, g_index):
    # A
    user_profile = generate_role_profile(g_index,meta_profile,  rel='self', user=None, name_str=None)
    name_str = '%s' % user_profile['姓名']

    relation_profiles = []
    relations = np.random.choice(meta_profile['亲属关系'],size=relation_size,replace=False)
    for r in relations:
        rp = generate_role_profile(g_index,meta_profile, rel=r,user=user_profile, name_str=name_str)
        name_str += '、%s' % rp['姓名']
        relation_profiles.append(rp)

    colleague_profiles = []
    relations = np.random.choice(meta_profile['同事关系'],size=colleague_size,replace=False)
    for r in relations:
        rp = generate_role_profile(g_index,meta_profile,  rel=r,user=user_profile, name_str=name_str)
        name_str += '、%s' % rp['姓名']
        colleague_profiles.append(rp)

    work_events = []
    event_types = np.random.choice(meta_profile['工作事件类型'], size=2, replace=False)
    for t in event_types:
        rp = generate_event_profile(meta_profile, t, user_profile)
        work_events.append(rp)

    rest_events = []
    event_types = np.random.choice(meta_profile['休闲事件类型'], size=2, replace=False)
    for t in event_types:
        rp = generate_event_profile(meta_profile, t, user_profile)
        rest_events.append(rp)

    for p in [user_profile] + relation_profiles + colleague_profiles:
        p['年龄'] = '%s岁' % p['年龄'] 
        p['身高'] = '%scm' % p['身高']

    items = []
    item_relations = np.random.choice(meta_profile['物品关系'], size=1, replace=False)
    for t in item_relations:
        rp = generate_item_profile(meta_profile, t, user_profile)
        items.append(rp)
    
    places = []
    place_relations = np.random.choice(meta_profile['地点关系'], size=1, replace=False)
    for t in place_relations:
        rp = generate_place_profile(meta_profile, t, user_profile)
        places.append(rp)

    graph_info = {
        'gid': g_index,
        'user_profile': user_profile,
        'relation_profiles': relation_profiles,
        'colleague_profiles': colleague_profiles,
        'work_events': work_events,
        'rest_events': rest_events,
        'items': items,
        'places': places
    }

    return graph_info


def generate_graphs():
    profiles_path = 'graphs.json'

    graph_list = []
    meta_profile = get_meta_profile()
    graph_num = len(meta_profile['姓名'])
    # graph_num = 3
    
    for g_index in range(graph_num):
        graph = generate_single_graph(meta_profile, g_index)
        print(graph)
        print(g_index)
        graph_list.append(graph)
    
    with open(profiles_path,'w', encoding='utf-8') as f:
        json.dump(graph_list, f, indent=4,ensure_ascii=False)


if __name__ == '__main__':
    generate_graphs()