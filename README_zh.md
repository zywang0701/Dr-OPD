<div align="center">

# Dr. OPD （On-policy Distillation Done Right)

### Learning What to Follow for Optimal<br>On-Policy Distillation of Large Language Models

[Zhenyu Wang](https://zywang0701.github.io/)<sup>\*</sup> · Tianze Wang<sup>\*</sup> · [Linjun Zhang](https://linjunz.github.io/) · [Yifan Hu](https://sites.google.com/view/yifan-hu)

Department of Statistics, Rutgers University<br>
<sub>\* 共同第一作者</sub>

<p>
  <a href="assets/paper.pdf"><img src="assets/badge-paper.svg" alt="阅读论文" height="30"></a>
  <a href="#快速开始"><img src="assets/badge-start.svg" alt="快速开始" height="30"></a>
  <a href="#实验结果"><img src="assets/badge-results.svg" alt="实验结果" height="30"></a>
</p>

[English](README.md) · **中文**

[方法介绍](#方法介绍) · [实验结果](#实验结果) · [快速开始](#快速开始) · [复现说明](docs/reproduction.md) · [引用](#引用)

</div>

## 方法介绍

On-policy distillation（OPD）在学生生成的回答上提供密集的 token-level 教师监督。Vanilla OPD 为每个教师信号赋予相同的权重；**Dr. OPD** 则根据这些信号对提升学生表现的帮助，自适应地调整权重。我们将其表述为：

$$
\max_w \left\lbrace\text{Student's performance after training with } w\text{-weighted OPD}\right\rbrace.
$$

其中，$w$ 是定义在所有可能生成的 token 上的权重函数。

精确求解 Dr. OPD 需要在不同的权重函数下反复训练学生，再选出最优权重，在计算上不可行。我们提出一个迭代求解器：每轮先以闭式解更新权重，再对所得的 weighted OPD 目标做一步梯度更新。

<p align="center">
  <a href="assets/figure1.png"><img src="assets/figure1.png" width="100%" alt="图 1：Vanilla OPD 使用相同权重；Dr. OPD 交替进行闭式权重更新与加权 OPD 梯度更新。"></a>
  <br><sub><b>图 1.</b> Dr. OPD 方法示意。</sub>
</p>

## 实验结果

- **在 strong-to-weak 蒸馏中**，Dr. OPD 在数学与代码任务上均优于所有比较的基线。相比 vanilla OPD，Qwen3-1.7B 的数学平均表现提升 **9.7 个百分点**，超过更大的 Qwen3-4B 教师（**35.0 vs. 34.1**）。
- **在 same-size 蒸馏中**，Dr. OPD 同样在所有比较的方法中取得最佳数学表现，并在 Instruct 和 Base 两种设置下均超过各自的 RL-trained teacher。

<p align="center">
  <a href="assets/figure2.png"><img src="assets/figure2.png" width="100%" alt="图 2：Dr. OPD 在图示的数学蒸馏设置中优于 vanilla OPD 和教师。"></a>
  <br><sub><b>图 2.</b> Strong-to-weak 与 same-size 蒸馏的实验结果。</sub>
</p>

图中数值为论文报告的五个数学 benchmark 的 **avg@16** 平均值；代码任务使用 **avg@4**。详见[完整结果与评测口径](docs/results.md)。

## 快速开始

当前仓库提供 **math 训练流程**、Dr. OPD 和四个基线。论文代码生成任务的结果见[结果文档](docs/results.md)，目前尚未包含对应的专用运行入口。

### 1. 查看配置，无需 GPU

在仓库根目录执行：

```bash
bash run.sh train dropd --dry-run
```

只打印启动配置，不创建环境、不查询 GPU、不启动 Ray、不下载模型，也不开始训练。

### 2. 准备环境和数据

论文实验使用单节点 **8 × H100 GPU**。原始代码面向 Linux、Python 3.11 和 CUDA 12.x；环境、模型及缓存预留约 60 GB，checkpoint 另计。尚未验证更低的最低 GPU 配置。

```bash
bash run.sh setup
```

设置准备好的 Parquet 文件路径，再检查数据行数：

```bash
export TRAIN_DATASET=/path/to/deepmath-level6-train.parquet
export TEST_DATASET=/path/to/valid_final_unique.parquet
bash run.sh data
```

数据格式与文件要求见[数据文档](docs/data.md)。本仓库不包含数据集二进制文件。

### 3. 训练

```bash
bash run.sh train dropd
```

| Student | Teacher | 权重强度 | 训练步数 | 数据随机种子 |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | Qwen3-4B | λ = 0.4 | 50 | 2 |

Checkpoint 选择与 benchmark 汇总方式详见[评测设置](docs/reproduction.md#evaluation)。

<details>
<summary><b>基线与参数覆盖</b></summary>

```bash
bash run.sh train plain
bash run.sh train grpd
bash run.sh train opdgrpo
bash run.sh train exopd
bash run.sh train dropd 0.4 2
STEPS=30 bash run.sh train dropd --dry-run
```

模型路径、输出位置及完整参数见[复现指南](docs/reproduction.md)。

</details>

## 代码导航

训练实现位于 `OPDVR/verl/`。

| 内容 | 文件 |
| --- | --- |
| Token weights | [`credit_gate.py`](OPDVR/verl/verl/workers/actor/credit_gate.py) |
| Token-level directional derivatives | [`jvp_influence.py`](OPDVR/verl/verl/workers/actor/jvp_influence.py) |
| Credits 计算 | [`dp_actor.py`](OPDVR/verl/verl/workers/actor/dp_actor.py) |
| 加权 OPD 训练 | [`ray_trainer.py`](OPDVR/verl/verl/trainer/ppo/ray_trainer.py) |

[方法细节](docs/method.md) · [复现指南](docs/reproduction.md) · [数据](docs/data.md) · [上游来源](UPSTREAM.md)

## 引用

```bibtex
@misc{wang2026dropd,
  title  = {Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models},
  author = {Wang, Zhenyu and Wang, Tianze and Zhang, Linjun and Hu, Yifan},
  year   = {2026},
  note   = {Preprint}
}
```

## 致谢

本实现基于 [OPD](https://github.com/thunlp/OPD) 与 [verl](https://github.com/volcengine/verl)，并使用了 OPDVR/GRPD 代码库的贡献。详见 [UPSTREAM.md](UPSTREAM.md) 和[第三方许可](LICENSE.md)。
