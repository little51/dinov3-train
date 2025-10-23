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

## 三、卫星图像分割

### 1、训练

```shell
# 1、激活环境
conda activate timm
# 2、安装matplotlib库
pip install matplotlib==3.10.6 -i https://pypi.mirrors.ustc.edu.cn/simple
# 3、设置Huggingface镜像
## Windows
set HF_ENDPOINT=https://hf-mirror.com
## Linux
export HF_ENDPOINT=https://hf-mirror.com
# 4、训练
python segment-train.py
```

### 2、测试

```shell
python segment-test.py
```

## 四、目标检测

### 1、训练集下载

```shell
# 下载地址
http://images.cocodataset.org/zips/train2017.zip 
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
http://images.cocodataset.org/zips/val2017.zip 
# 目录结构
data/coco/
├── train2017/          # 训练图像
├── val2017/            # 验证图像
└── annotations/        # 标注
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 2、训练

```shell
# 1、激活环境
conda activate timm
# 2、安装pycocotools库
pip install pycocotools==2.0.10 -i https://pypi.mirrors.ustc.edu.cn/simple
# 3、设置Huggingface镜像
## Windows
set HF_ENDPOINT=https://hf-mirror.com
## Linux
export HF_ENDPOINT=https://hf-mirror.com
# 4、训练
python detection-train.py
```

### 3、测试

```shell
python detection-test.py
```

## 五、自定义数据集分类训练

```
# 1、激活环境
conda activate timm
# 2、设置Huggingface镜像
## Windows
set HF_ENDPOINT=https://hf-mirror.com
## Linux
export HF_ENDPOINT=https://hf-mirror.com
# 3、训练
python mydataset-train.py
```

## 六、蒸馏训练模型

```shell
# 1、数据准备
git clone https://github.com/lightly-ai/dataset_clothing_images.git my_data_dir
# 删除数据目录下的.git目录
# Linux执行以下命令，Windows上直接删除.git目录
rm -rf my_data_dir/.git
# 2、安装lightly-train库
conda activate timm
pip install lightly-train -i https://pypi.mirrors.ustc.edu.cn/simple
# 3、蒸馏
python lightly-train-dinov3.py
# 4、微调
python lightly-train-finetun.py
# 5、分类
python lightly-train-infer.py
```



## 详细原理、源码解析和操作步骤介绍，敬请关注作者公众号。

![](https://gitclone.com/download1/aliendao/weixin-aliendao2.jpg)