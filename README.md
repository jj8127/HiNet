# HiNet: Deep Image Hiding by Invertible Network
This repository contains the official implementation of the paper [**HiNet: Deep Image Hiding by Invertible Network**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html) from ICCV&nbsp;2021 by [MC2 Lab](http://buaamc2.net/) at Beihang University.

![Architecture](HiNet.png)

## Dependencies and Installation
- Python 3 (recommend using [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch >= 2.7.1](https://pytorch.org/).
- See `environment_torch2.yml` for an example conda environment using the CUDA&nbsp;11.8 wheels.

### Using conda
```bash
conda env create -f environment_torch2.yml
conda activate hinet_pytorch2
```

### Using pip (CUDA build)
```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  torchaudio==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Using pip (CPU build)
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Make sure that `torch` and `torchvision` use matching versions and build types (both CPU or both CUDA).

## Get Started
- Train a model with `python train.py`.
- Test a model with `python test.py`.
- Adjust dataset and output paths in `config.py` to match your setup.

A pre-trained model is available [here](https://drive.google.com/drive/folders/1l3XBFYPMaNFdvCWyOHfB2qIPkpjIxZgE?usp=sharing).

## Partial INT8 Quantization
The script `qat_partial.py` demonstrates quantization aware training of the convolution layers only. After calibration it saves a quantized model which runs on CPU.

Run the example:
```bash
python qat_partial.py --pretrained /path/to/model.pt --epochs 5 --calib-steps 10
```

## License
This project is licensed under the [MIT License](LICENSE).

## Citation
If you use this work, please cite:
```bibtex
@InProceedings{Jing_2021_ICCV,
    author    = {Jing, Junpeng and Deng, Xin and Xu, Mai and Wang, Jianyi and Guan, Zhenyu},
    title     = {HiNet: Deep Image Hiding by Invertible Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4733-4742}
}
```
