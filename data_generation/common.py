# This file is the common function for all generations.

import numpy as np

from utils import remove_space_and_ent, create_LLM


llm = create_LLM({
    'model_name': 'GLM-4-0520',
    'model_type': 'remote',
    'api_key': 'XXX-API-KEY'
})

def rewrite_message(message, charact):
    """
    The characteristic attr has been discarded here.
    """
    prompt = '[用户消息] %s \n' % message
    prompt += '请用口语化的陈述句重写上述用户消息，保证通顺且没有语病，不要改变原有信息，且不涉及“你”的内容。'
    prompt += '只输出重写后的用户消息，不要输出原有的用户消息，不要输出其他描述信息。\n'
    prompt += '输出示例：我爸爸的生日是5月29日。'
    return remove_space_and_ent(llm.fast_run(prompt))

def rewrite_message_event(message, charact):
    """
    The characteristic attr has been discarded here.
    """
    return message

def rewrite_message_role(message, charact):
    """
    The characteristic attr has been discarded here.
    """
    return message

def rewrite_question(question):
    prompt = '[问题] %s \n' % question
    prompt += '请用口语化的问句重写上述问题，保证通顺且没有语病，不要改变原有信息，且不涉及“你”的内容。'
    prompt += '只输出重写后的问题，不要输出原有的问题，不要输出其他描述信息。\n'
    prompt += '输出示例：我表哥的生日是哪一天？'
    return remove_space_and_ent(llm.fast_run(prompt))

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

def formulate_QA(question, answer, other_answers=None):
    if other_answers:
        groud_truth,choices = get_choices(answer,other_answers)
        return question, choices, groud_truth

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

    prompt = '问题： %s\n' % question
    prompt += '正确答案: %s\n' % answer
    prompt += '请帮我根据上述问题和答案，生成三个不同的混淆选项。\n'
    prompt += '每一个混淆选项需要和答案的内容和长度相似，可以有部分文字重叠。如果正确答案是一串数字，你可以修改其中的1到3位。\n'
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

    groud_truth,choices = get_choices(answer,other_answers)
    return question, choices, groud_truth


def formulate_QA_additional_judge(question, answer, other_answers=None):
    if other_answers:
        groud_truth,choices = get_choices(answer,other_answers)
        return question, choices, groud_truth

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

    prompt = '问题： %s\n' % question
    prompt += '正确答案: %s\n' % answer
    prompt += '请帮我根据上述问题和答案，生成三个不同的混淆选项，每个混淆选项与答案都是矛盾冲突的。\n'
    prompt += '每一个混淆选项需要和答案的内容和长度相似，可以有部分文字重叠。\n'
    prompt += '输出应该遵循以下示例格式：\n'
    prompt += 'A. 价格低廉，外观美观，性价比高\n'
    prompt += 'B. 价格昂贵，外观不美观，性价比较低\n'
    prompt += 'C. 充电慢，不耐用，性价比较低\n'

    res = llm.fast_run(prompt)
    other_answers = get_other_answers(res)
    max_tries = 10
    while not other_answers:
        res = llm.fast_run(prompt)
        max_tries -= 1
        other_answers = get_other_answers(res)
        if max_tries <= 0:
            other_answers = ['不知道', '都不正确', '未提及答案']

    groud_truth,choices = get_choices(answer,other_answers)
    return question, choices, groud_truth