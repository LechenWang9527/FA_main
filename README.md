# FA-Neg-CG-CAM

这是一个基于 CLIP 的少样本图像分类与分布外检测研究项目。项目在 Forced Prompt Learning（FA）框架上加入负提示学习，并实现置信度门控和类别自适应间隔，用于探索模型对已知类别与分布外样本的区分能力。

## 方法简介

- 使用 CLIP 图像编码器与可学习提示进行少样本分类。
- 加入负提示分支，为类别构造互补的负向监督。
- 训练时使用置信度门控，并按类别调整间隔参数。
- 评估已知类别分类表现，以及分布外检测的 AUROC、AUPR 和 FPR@95。

## 目录结构

- `main.py`：训练与评估入口
- `model.py`：提示学习模型及相关网络组件
- `configs/`：实验配置文件
- `my_dataset/`：数据集加载代码
- `ood_utils/`：分布外检测评估指标
- `utils.py`：训练与评估辅助函数
- `train.sh`：示例训练和测试命令

## 运行

安装 `requirements.txt` 中的依赖，并准备相应数据集与 CLIP 预训练权重。数据和权重未包含在仓库中。运行前请在配置文件中修改数据目录、权重目录及实验参数。

```bash
python main.py --config configs/eurosat_dtd_faneg_correctgate_cam.yaml --is_train 1
python main.py --config configs/eurosat_dtd_faneg_correctgate_cam.yaml --is_train 0
```

`--is_train 1` 执行训练，`--is_train 0` 执行评估。具体数据集目录结构需符合 `my_dataset/` 中的数据加载实现。