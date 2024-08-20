# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.
import argparse
import os

import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    model: str
    res_message: str


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    temperature = 0.95 if not request.temperature else request.temperature
    top_p = 0.9 if not request.top_p else request.top_p

    messages = request.messages
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    gen_kwargs = {"max_length": 128000, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**input_ids, **gen_kwargs)
        outputs = outputs[:, input_ids['input_ids'].shape[1]:]
        res_message = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return ChatCompletionResponse(model=request.model, res_message = res_message)


if __name__ == "__main__":
    LLM_config = {
        'model_path': '/data/zhangzeyu/local_llms/glm-4-9b-chat',
        'model_name': 'glm-4-9b-chat',
#        'usable_port': [8006, 8007],
        'usable_port': [8086, 8087],
        'usable_gpu': [3,5,6]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('index', default=0, type=int)
    # parser.add_argument('port', default=8006, type=int)
    args, extra_args = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(LLM_config['usable_gpu'][args.index])
    current_port = LLM_config['usable_port'][args.index]

    tokenizer = AutoTokenizer.from_pretrained(LLM_config['model_path'], trust_remote_code=True, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_config['model_path'], trust_remote_code=True,torch_dtype=torch.float16).cuda()

    print('LLM: %s with path %s on CUDA %d at port %d.' % (LLM_config['model_name'], LLM_config['model_path'], LLM_config['usable_gpu'][args.index],current_port))
    uvicorn.run(app, host='0.0.0.0', port=current_port, workers=1)