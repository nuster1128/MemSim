from transformers import AutoModel, AutoTokenizer
import faiss
import torch
from methods.BaseAgent import BaseAgent
from utils import create_LLM, remove_space_and_ent

class RetMemAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        model_path = self.config['embedding_config']['model_path']
        self.embedding_dim = self.config['embedding_config']['embedding_dim']
        self.top_k = self.config['top_k']

        self.llm = create_LLM(config['LLM_config'])

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        self.memorystore = []
        self.vectorstore = faiss.IndexFlatIP(self.embedding_dim)

    def __convert_strings_to_vectors__(self,s):
        res = self.tokenizer(s, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**res).last_hidden_state[:, -1, :]
            norms = torch.linalg.norm(embeddings,dim=1,keepdim = True)
            embedding_normalized = embeddings / norms

        return embedding_normalized.numpy()

    def __write_memory__(self, mem):
        self.memorystore.append(mem)
        mem_encode = self.__convert_strings_to_vectors__(mem)
        self.vectorstore.add(mem_encode)

    def __read_memory__(self, query, topk = None):
        if not topk:
            topk = self.top_k
        query_emb = self.__convert_strings_to_vectors__(query)
        dis, idx = self.vectorstore.search(query_emb,min(len(self.memorystore), topk))
        mem = [self.memorystore[i] for i in idx[0]]
        
        mem_str = ''
        for index, m in enumerate(mem):
            mem_str += '[%d] %s\n' % (index, m)
        return mem_str, list(idx[0])

    def reset(self, **kwargs):
        self.vectorstore = faiss.IndexFlatIP(self.embedding_dim)
        self.memorystore = []

    def observe_without_action(self, obs):
        self.__write_memory__(obs)

    def response_answer(self, question, choices, time):
        prompt = '[用户消息] \n'
        prompt += self.__read_memory__(question)[0]
        prompt += '[单选题] %s \n' % question
        for k,v in choices.items():
            prompt += '%s: %s\n' % (k,v)
        prompt += '当前时间是 %s\n' % time
        prompt += '请根据[用户消息]，给出[单选题]的正确答案。\n'
        prompt += '只输出答案所对应的选项，不要输出解释或其他任何的内容。\n'
        prompt += '输出样例: A'

        res = remove_space_and_ent(self.llm.fast_run(prompt))

        # print(prompt)
        # print(res)

        return res

    def response_retri(self, question, topk = 5):
        return self.__read_memory__(question, topk)[1]
    
    def process(self):
        pass

