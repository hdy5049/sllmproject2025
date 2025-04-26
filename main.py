from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 데이터 구조
class TextInput(BaseModel):
    text: str

# Hugging Face Access Token
hf_token = " "

# 모델 설정
model_name = "openchat/openchat_3.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# 텍스트 생성 API
@app.post("/generate")
async def generate_text(input_data: TextInput):
    inputs = tokenizer(input_data.text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}
