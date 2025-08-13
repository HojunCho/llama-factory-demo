#!/usr/bin/env python3
"""
Hugging Face의 mdhasnainali/job-html-to-json 데이터셋을 다운로드하고
llama-factory 형식에 맞는 train.json을 생성하는 스크립트
"""

import json
import os
from typing import List, Dict, Any
from datasets import load_dataset
from pydantic import BaseModel


class ApplicationInfo(BaseModel):
    apply_url: str
    contact_email: str
    deadline: str


class Location(BaseModel):
    city: str
    state: str | None
    country: str | None
    remote: bool | None
    hybrid: bool | None


class Qualifications(BaseModel):
    education_level: str | None
    fields_of_study: str | None
    certifications: list[str] | None


class Salary(BaseModel):
    currency: str | None
    min: float | None
    max: float | None
    period: str | None


class YearsOfExperience(BaseModel):
    min: float | None
    max: float | None


class JobPosting(BaseModel):
    job_id: str
    title: str
    department: str
    employment_type: str
    experience_level: str
    posted_date: str
    work_schedule: str

    location: Location
    application_info: ApplicationInfo
    salary: Salary
    years_of_experience: YearsOfExperience

    requirements: list[str]
    responsibilities: list[str]
    nice_to_have: list[str]
    qualifications: Qualifications
    recruitment_process: list[str]
    programming_languages: list[str]
    tools: list[str]
    databases: list[str]
    cloud_providers: list[str]
    language_requirements: list[str]
    benefits: list[str]


def create_system_prompt() -> str:
    """formatting.ipynb에서 사용된 system prompt를 생성"""
    json_schema = json.dumps(JobPosting.model_json_schema(), indent=2)
    
    system_prompt = f"""주어진 HTML 입력을 JSON으로 바꾸시오. 이때 JSON schema는 다음과 같다:
{json_schema}

HTML에서 정보를 추출하여 위 스키마에 맞는 JSON 형태로 변환하시오. 다른 텍스트는 포함하지 말고 json을 바로 출력하시오."""
    
    return system_prompt


def download_dataset():
    """Hugging Face에서 데이터셋 다운로드"""
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset("mdhasnainali/job-html-to-json", split="train")
    print(f"Dataset loaded with {len(dataset)} examples")
    return dataset


def convert_to_llama_factory_format(dataset, max_samples: int = None) -> List[Dict[str, Any]]:
    """데이터셋을 llama-factory 형식으로 변환"""
    system_prompt = create_system_prompt()
    
    llama_factory_data = []
    
    # 샘플 수 제한
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Converting {len(dataset)} samples to llama-factory format...")
    
    for i, item in enumerate(dataset):
        try:
            # HTML 입력
            html_input = item.get('html', '').strip("\"")
            
            # JSON 출력 (문자열로 저장되어 있을 수 있음)
            json_output = item.get('json', '')
            
            # JSON이 문자열로 저장되어 있다면 파싱
            if isinstance(json_output, str):
                try:
                    json_output = json.loads(json_output)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON for sample {i}")
                    continue
            
            # JSON을 다시 문자열로 변환 (정렬된 형태로)
            json_output_str = json.dumps(json_output, ensure_ascii=False, indent=2)
            
            # llama-factory 형식으로 변환
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": html_input
                    },
                    {
                        "role": "assistant",
                        "content": json_output_str
                    }
                ]
            }
            
            llama_factory_data.append(conversation)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    print(f"Successfully converted {len(llama_factory_data)} samples")
    return llama_factory_data


def save_train_json(data: List[Dict[str, Any]], output_path: str):
    """train.json 파일로 저장"""
    print(f"Saving data to {output_path}...")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(data)} conversations to {output_path}")


def main():
    """메인 실행 함수"""
    # 데이터셋 다운로드
    dataset = download_dataset()
    
    # 처음 몇 개 샘플 확인
    print("\nFirst sample structure:")
    print("Keys:", list(dataset[0].keys()))
    
    # 첫 번째 샘플의 구조 출력
    first_sample = dataset[0]
    print("\nFirst sample preview:")
    for key, value in first_sample.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"{key}: {value[:200]}...")
        else:
            print(f"{key}: {value}")
    
    # llama-factory 형식으로 변환 (처음 1000개 샘플만 사용)
    llama_factory_data = convert_to_llama_factory_format(dataset)
    
    # train.json으로 저장
    output_path = "data/formatting/train.json"
    save_train_json(llama_factory_data, output_path)
    
    print("\nConversion completed!")
    print(f"Output file: {output_path}")
    print(f"Total conversations: {len(llama_factory_data)}")


if __name__ == "__main__":
    main()
