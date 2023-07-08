## <div align="center"><b>Restore Video by Real-ESRGAN</b></div>

## Introduction

This project is based on the open source project Real-ESRGAN(https://github.com/xinntao/Real-ESRGAN).
It is able to achieve the repair of both realistic videos and anime videos.

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

## Demos Videos

#### Bilibili

- [Real-ESRGAN Native Video Demo](https://www.bilibili.com/video/BV17V4y1h7KW/)

## Installation

1. Clone repo

    ```bash
    git clone https://github.com/KeYaGuSh/Restore_Video_by_Real_ESRGAN.git
    cd Restore_Video_by_Real_ESRGAN
    ```

2. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    # facexlib and gfpgan are for face enhancement
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

---

3. Run

    ```bash
    python video_restoration.py
    ```

4. Models
    
    Two models are included.
        1.RealESRGAN_x4plus(for realistic video)
        2.realesr-animevideov3(for anime video)

## Attention

    # It can only restore mp4 videos!!!
    # If the format of your videos is not mp4, you can use the video format converter to convert the video format.
    python video_format_converter.py

## Contact

If you have any question, please email '2935926294@qq.com'.
