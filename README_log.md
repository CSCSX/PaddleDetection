# 说明文档

## 运行方法

> 先根据官方文档配好环境

- 训练

```powershell
python .\tools\train.py -c .\configs\faster_rcnn\tmp.yaml -o use_gpu=true 
```

- 测试（一张图片）结果在 `output` 目录下

```powershell
python tools/infer.py -c .\configs\faster_rcnn\customed.yaml -o use_gpu=true --infer_img=demo/hrnet_demo.jpg
```

## 变动内容

### v0

- 新增 `configs\faster_rcnn\customed.yaml` 配置文件
- 修改 `ppdet\modeling\architectures\__init__.py` 中增加了两行
  - `from . import architecture_new`
  - `from .architecture_new import *`
- 新增 `ppdet\modeling\architectures\architecture_new.py` 文件作为**主要模块**
  - `_forward` 方法中实现了主要逻辑
- 修改 `ppdet\modeling\backbones\resnet.py` 中增加了 `ResNet50` 和 `ResNet101` 两个类

### v1

- 修改 `ppdet\modeling\architectures\architecture_new.py` 新增了 `FixedPatchPrompter_image` 模块，添加了 `backbone` 出来的特征之间的 L1 距离

## 存在问题

由于该框架的抽象程度很高，个人感觉实现将两个 `FasterRCNN` 放在一个工作流里的实现很受约束，于是在一些地方打破了抽象层，比如在 `ppdet\modeling\architectures\architecture_new.py` 中夹杂了权重文件的预处理，在 `ppdet\modeling\backbones\resnet.py` 中细分了 `ResNet50` 和 `ResNet101` 两个类
