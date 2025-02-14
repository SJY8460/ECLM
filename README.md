The official code of paper ECLM.
# 项目名称

本项目提供了用于训练和推断的脚本，基于 Mistral-7B-Instruct-v0.1 模型。你可以通过下面的命令对模型进行微调训练以及推断测试。

## 环境要求

- Python 3.x
- PyTorch（请根据实际情况安装合适的版本）
- 其他依赖请参考 `requirements.txt`（如果有）

## 训练

使用以下命令启动模型训练：

```bash
python train.py -md "../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1" --template_type sub -bs 4
