# DINOV3训练篇

## 一、环境准备

```shell
# 1、创建虚拟环境
conda create -n timm python=3.12 -y
# 2、激活虚拟环境
conda activate timm
# 3、安装timm及其他依赖库
pip install timm==1.0.20 -i https://pypi.mirrors.ustc.edu.cn/simple
# 4、重装PyTorch（Windows）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 5、验证是否安装成功
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、图像分类

### 1、训练

```shell
# 1、激活环境
conda activate timm
# 2、设置Huggingface镜像
## Windows
set HF_ENDPOINT=https://hf-mirror.com
## Linux
export HF_ENDPOINT=https://hf-mirror.com
# 3、训练
python classifier-train.py
```

### 2、测试

```shell
python classifier-test.py
```

详细原理、源码解析和操作步骤介绍，敬请关注作者公众号。

![](https://gitclone.com/download1/aliendao/weixin-aliendao2.jpg)

