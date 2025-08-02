# HiNet: 역변환 네트워크를 이용한 이미지 은닉

이 저장소는 ICCV 2021에 발표된 **HiNet: Deep Image Hiding by Invertible Network**의 공식 구현입니다. 역변환 가능한 네트워크를 사용하여 비밀 이미지를 커버 이미지에 숨기고 다시 복원할 수 있는 모델을 제공합니다.

<center>
  <img src="https://github.com/TomTomTommi/HiNet/blob/main/HiNet.png" width="60%" />
</center>

## 환경 및 설치
- Python 3 (권장: [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 2.7.1 + CUDA 11.8](https://pytorch.org/)
- 예시 환경은 `environment_torch2.yml`에 제공됩니다.

```bash
conda env create -f environment_torch2.yml
conda activate hinet_pytorch2
```

## 기본 사용법

### 1. `config.py` 설정
모든 하이퍼파라미터와 경로가 `config.py`에 정의되어 있습니다.
- `TRAIN_PATH`, `VAL_PATH`: 비밀 이미지 데이터 경로
- `TRAIN_COVER_PATH`, `VAL_COVER_PATH`: 커버 이미지 경로
- `MODEL_PATH`: 학습된 모델 저장 위치
- `IMAGE_PATH`: 테스트 시 생성되는 이미지 저장 위치
- `batch_size`, `epochs`, `lamda_reconstruction` 등의 학습 파라미터를 필요에 따라 조정합니다.

### 2. 학습
```bash
python train.py
```

### 3. 평가
```bash
python test.py
```
테스트 전에 `MODEL_PATH`와 `IMAGE_PATH`를 올바른 경로로 수정해야 합니다.

## 양자화 학습(QAT)

### 8비트 QAT `qat_8bit.py`
`nn.Conv2d` 계층만 8비트로 양자화하고 `INV_block`은 FP32로 유지하는 부분 양자화를 수행합니다. 학습과 보정(calibration) 후에는 양자화된 모델이 `model/model_qat_ep{EPOCHS}_calib{STEPS}.pt`로 저장됩니다.

```bash
python qat_8bit.py --pretrained /path/to/model.pt \
                   --epochs 5 --calib-steps 10
```

### 4비트 QAT `qat_4bit.py`
4비트 fake quantization을 적용한 QAT를 수행합니다. 사용 방법은 8비트 버전과 동일하며, 결과는 `model/model_qat4bit_ep{EPOCHS}_calib{STEPS}.pt`로 저장됩니다.

```bash
python qat_4bit.py --pretrained /path/to/model.pt \
                   --epochs 5 --calib-steps 10
```

### 양자화된 모델 사용
- `demo_quantized.py`: 양자화된 모델을 이용해 예시 이미지를 저장합니다.
  ```bash
  python demo_quantized.py --model model/model_qat_ep5_calib10.pt
  ```
- `run_quantized.py`: 테스트 세트에서 PSNR/SSIM을 계산하고 스테고/복원 이미지를 저장합니다.
  ```bash
  python run_quantized.py --model model/model_qat_ep5_calib10.pt
  ```

## 기타
- `batchsize_val`은 GPU 수의 두 배 이상이며 GPU 수로 나누어떨어져야 합니다.
- 사용자 데이터셋을 사용하려면 `config.py`의 경로를 원하는 위치로 변경하십시오.

## 인용
연구나 코드가 도움이 되었다면 다음을 인용해 주세요:
```text
@InProceedings{Jing_2021_ICCV,
    author    = {Jing, Junpeng and Deng, Xin and Xu, Mai and Wang, Jianyi and Guan, Zhenyu},
    title     = {HiNet: Deep Image Hiding by Invertible Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4733-4742}
}
```

