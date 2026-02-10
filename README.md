# PyTorch Image Classification Baseline

MNIST 데이터셋을 사용한 이미지 분류 베이스라인 프로젝트입니다.

## Features

- **모델**: CNN, MLP 지원
- **재현성**: Seed 고정으로 재현 가능한 실험
- **로깅**: Training history 및 metrics 자동 저장
- **시각화**: Loss/Accuracy 그래프 자동 생성
- **Best Model**: Validation accuracy 기준 최고 모델 저장 옵션

## Requirements

- Python 3.12
- PyTorch 2.10.0
- torchvision 0.25.0
- numpy 2.4.2
- matplotlib 3.10.8

## Installation

```bash
# 의존성 설치
pip install -r requirements.txt
```

## Usage

### 기본 학습 (CNN 모델)

```bash
python train.py
```

### MLP 모델 학습

```bash
python train.py --model mlp
```

### 고급 옵션

```bash
python train.py --model cnn \
                --epochs 10 \
                --batch-size 256 \
                --lr 0.001 \
                --seed 42 \
                --save-best
```

### 인자 설명

- `--model`: 모델 타입 (cnn, mlp) [기본값: cnn]
- `--epochs`: 학습 에포크 수 [기본값: 5]
- `--batch-size`: 배치 크기 [기본값: 128]
- `--lr`: Learning rate [기본값: 0.001]
- `--seed`: Random seed [기본값: 42]
- `--save-best`: Validation accuracy 기준 최고 모델 저장

## Project Structure

```
pytorch-image-classification-baseline/
├── data.py              # 데이터 로딩 및 전처리
├── model.py             # 모델 정의 (SimpleCNN, SimpleMLP)
├── train.py             # 학습 스크립트
├── utils.py             # 유틸리티 함수
├── requirements.txt     # 의존성 목록
├── data/                # 데이터셋 (자동 다운로드)
└── outputs/             # 학습 결과 저장
    └── YYMMDD_HHMMSS/   # 타임스탬프별 결과
        ├── metrics.json     # 학습 메트릭
        ├── history.png      # Loss/Accuracy 그래프
        └── best_model.pt    # 최고 모델 (--save-best 사용 시)
```

## Output Format

### metrics.json

```json
{
  "meta": {
    "model": "cnn",
    "epochs": 5,
    "batch_size": 128,
    "lr": 0.001,
    "seed": 42,
    "device": "cuda",
    "val_rate": 0.1,
    "start_time": "2026-02-10T13:33:13",
    "end_time": "2026-02-10T13:35:20",
    "duration_sec": 127.456
  },
  "metrics": {
    "test_loss": 0.0234,
    "test_acc": 0.9912,
    "history": {
      "train_loss": [...],
      "train_acc": [...],
      "val_loss": [...],
      "val_acc": [...]
    },
    "best": {
      "enabled": true,
      "epoch": 4,
      "val_acc": 0.9895,
      "path": "outputs/260210_133313/best_model.pt"
    }
  }
}
```

## Model Architecture

### SimpleCNN
- Conv2d(1→32) + ReLU + MaxPool
- Conv2d(32→64) + ReLU + MaxPool
- FC(3136→128) + ReLU
- FC(128→10)

### SimpleMLP
- FC(784→128) + ReLU
- FC(128→10)

## Results

MNIST 데이터셋에서의 예상 성능:
- **CNN**: ~99% test accuracy (5 epochs)
- **MLP**: ~97% test accuracy (5 epochs)

