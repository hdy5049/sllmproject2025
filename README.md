# sllmproject2025
캡스톤프로젝트2 과목 세번째 프로젝트, sLLM 만들어보기

https://huggingface.co/google/gemma-2-2b-it/tree/main
# 🤖 홍도영의 인공지능 서비스 프로젝트

# 1. 프로젝트 개요

홍도영의 인공지능 서비스는 FastAPI와 HuggingFace 모델을 활용하여, 사용자가 입력한 질문에 자연스러운 답변을 생성하는 웹 기반 인공지능 애플리케이션입니다.본 프로젝트는 경량화된 언어모델을 실서비스에 적용하고, 다양한 입력에 대해 일관성 있는 답변을 제공하는 것을 목표로 합니다.

# 2. 사용 모델

모델명: openchat/openchat_3.5 (추가 테스트: google/gemma-2-2b-it)

특징:

다양한 대화형 데이터로 학습된 고성능 언어모델

GPU 환경 최적화 지원 (float16 연산)

자유로운 자연어 질의응답 지원

# 3. 시스템 개요

서버 측

FastAPI로 RESTful API 서버 구축

Torch + Transformers를 통해 모델 로딩 및 추론 처리

CORS 설정으로 프론트엔드 통신 허용

클라이언트 측

HTML + Bootstrap + JavaScript 기반 간단한 사용자 인터페이스

입력 폼 및 답변 출력

답변 생성 시간 측정 및 출력 기능 포함

# 4. 구현 화면



# 5. 핵심 코드 요약

## 1. 모델 로드 (GPU 사용)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "openchat/openchat_3.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

## 2. 질문 입력 및 답변 생성

inputs = tokenizer("질문 내용", return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

## 3. FastAPI 서버 기본 구조

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(input_data: TextInput):
    return {"response": "생성된 답변"}

# 6. 세부 기술적 특징

항목

내용

서버 프레임워크

FastAPI

추론 프레임워크

PyTorch + HuggingFace Transformers

디바이스 설정

GPU 사용 (CUDA) / CPU fallback 지원

최적화

float16 연산 최적화 (GPU 사용 시)

프론트엔드

HTML5, CSS3, Bootstrap5, Vanilla JavaScript

# 7. 느낀 점 및 개선 방향

장점:

간결한 구조로 빠른 개발 가능

오픈소스 모델 사용으로 유연성 확보

발견된 문제점:

답변 일관성 부족 (특히 긴 질문)

짧은 응답 시간이 필요한 경우 성능 튜닝 필요

개선 아이디어:

Prompt Template 고도화

추가적인 post-processing으로 문장 정제

한글 성능 향상을 위한 fine-tuning 적용 고려
