# GORetrieverApp

GUI for GORetriever using pyqt6, cloned from https://github.com/ZhuLab-Fudan/GORetriever/

# Installing

Get python 3.8 and java jdk 21

Create python3.8 virtual environment, then install:
```
pip install -r requirements.txt --no-deps
```
(Use requirements-windows.txt for windows)

Download cross_encoder from https://drive.google.com/file/d/11W51FnM62Z79qGPkuZHRzAv6Bx_L1Mah/view?usp=sharing into folder **cross_model**


For GPU usage: NVIDIA CUDA GPU needed, download CUDA 11 and pytorch version:
https://developer.nvidia.com/cuda-11.0-download-archive
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Running

```
python predict.py
```

results will be in results/
cached files in test/
