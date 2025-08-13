"""
Intel Image Classification Dataset Converter for LlamaFactory
이 스크립트는 Hugging Face의 intel-image-classification 데이터셋을 
LlamaFactory 형식으로 변환합니다.
"""

import os
import json
from datasets import load_dataset
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_split(split_name):
    """
    특정 split(train 또는 test)을 처리하는 함수
    """
    # 경로 설정
    data_dir = os.path.join("data", "image_classification")
    images_dir = os.path.join(data_dir, "images")
    json_path = os.path.join(data_dir, f"{split_name}.json")
    
    # 이미지 저장 디렉토리 생성
    os.makedirs(images_dir, exist_ok=True)
    
    logger.info(f"Intel Image Classification 데이터셋 ({split_name}) 다운로드 중...")
    
    try:
        # Hugging Face 데이터셋 로드
        dataset = load_dataset("sfarrukhm/intel-image-classification", split=split_name)
        logger.info(f"{split_name} 데이터셋 로드 완료: {len(dataset)}개의 샘플")
        
        # 라벨 매핑 (데이터셋의 라벨을 확인)
        # 일반적으로 intel image classification의 클래스들
        label_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
        
        # LlamaFactory 형식으로 변환할 데이터 리스트
        converted_data = []
        
        logger.info(f"{split_name} 데이터 변환 및 이미지 저장 시작...")
        
        for idx, sample in enumerate(dataset):
            # 이미지 저장 (split별로 파일명 구분)
            image = sample["image"]
            image_filename = f"{split_name}_{idx}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # 이미지를 RGB 모드로 변환 후 저장
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_path, "JPEG", quality=95)
            
            # 라벨 이름 가져오기
            label_id = sample["label"]
            label_name = label_names[label_id] if label_id < len(label_names) else f"class_{label_id}"
            
            # LlamaFactory 형식으로 변환
            converted_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "주어진 이미지에 대해서 다음 중 하나로 분류하시오: ['Mountain', 'Glacier', 'Street', 'Sea', 'Forest', 'Buildings']"
                    },
                    {
                        "role": "user",
                        "content": "<image>"
                    },
                    {
                        "role": "assistant",
                        "content": label_name.capitalize()
                    }
                ],
                "images": [
                    f"image_classification/images/{image_filename}"
                ]
            }
            
            converted_data.append(converted_sample)
            
            # 진행 상황 로그
            if (idx + 1) % 1000 == 0 or idx == 0:
                logger.info(f"{split_name} 처리 완료: {idx + 1}/{len(dataset)} 샘플")
        
        # JSON 파일 저장
        logger.info(f"{split_name}.json 파일 저장 중: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{split_name} 변환 완료!")
        logger.info(f"- 총 {len(converted_data)}개의 샘플 변환")
        logger.info(f"- 이미지 저장 위치: {images_dir}")
        logger.info(f"- {split_name}.json 저장 위치: {json_path}")
        
        return converted_data
        
    except Exception as e:
        logger.error(f"{split_name} 처리 중 오류 발생: {str(e)}")
        raise

def download_and_convert_dataset():
    """
    Intel Image Classification 데이터셋을 다운받아 LlamaFactory 형식으로 변환
    """
    try:
        # train split 처리
        train_data = process_split("train")
        
        # test split 처리
        test_data = process_split("test")
        
        logger.info("전체 변환 완료!")
        logger.info(f"- Train 샘플: {len(train_data)}개")
        logger.info(f"- Test 샘플: {len(test_data)}개")
        
        # 샘플 데이터 출력
        if train_data:
            logger.info("Train 샘플 데이터:")
            print(json.dumps(train_data[0], ensure_ascii=False, indent=2))
        
        if test_data:
            logger.info("Test 샘플 데이터:")
            print(json.dumps(test_data[0], ensure_ascii=False, indent=2))
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise

def verify_dataset():
    """
    변환된 데이터셋 검증
    """
    data_dir = "data/image_classification"
    images_dir = os.path.join(data_dir, "images")
    
    for split_name in ["train", "test"]:
        json_path = os.path.join(data_dir, f"{split_name}.json")
        
        if not os.path.exists(json_path):
            logger.error(f"{split_name}.json 파일이 존재하지 않습니다.")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"{split_name.upper()} 검증 결과:")
        logger.info(f"- {split_name}.json 샘플 수: {len(data)}")
        
        # 해당 split의 이미지 파일 개수 확인
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.startswith(f"{split_name}_") and f.endswith('.jpg')]
            logger.info(f"- {split_name} 이미지 파일 수: {len(image_files)}")
        
        # 라벨 분포 확인
        labels = {}
        for sample in data:
            label = sample["messages"][2]["content"]
            labels[label] = labels.get(label, 0) + 1
        
        logger.info(f"{split_name} 라벨 분포:")
        for label, count in sorted(labels.items()):
            logger.info(f"  {label}: {count}개")
        logger.info("")

if __name__ == "__main__":
    print("Intel Image Classification Dataset Converter")
    print("=" * 50)
    
    # 데이터셋 다운로드 및 변환
    download_and_convert_dataset()
    
    # 결과 검증
    print("\n" + "=" * 50)
    verify_dataset()
