# DINO-X的开源版本本grounding-dino测试

## 权重下载

```shell
https://hf-mirror.com/IDEA-Research/grounding-dino-base/tree/main
```

## 目标检测
```shell
# 建立虚拟环境
conda create -n dinox python=3.12 -y
# 激活虚拟环境
conda activate dinox
# 安装依赖库
pip install torch requests transformers pillow -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装Pytorch(Windows)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 运行检测例徎
python grounding-dino.py
```

