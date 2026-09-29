# 基于 Hypnos 的睡眠基础模型改进方案调研

> **目标**：在 Hypnos 基础上筛选可发表于顶会（ICML / NeurIPS / ICLR）的创新点。
> **约束**：复用 Hypnos 的两个核心概念：(1) per-modality tokenizer；(2) per-second motif。
> **范围**：从 Hypnos 已借鉴的语音、文本领域，以及视觉、时间序列、生物序列建模中，引入与 Hypnos 有差异、且有潜力成为创新点的方案，逐一评估创新性与可行性，并给出参考文献与可行性证据。
> **日期**：2026-09-28

**证据来源说明**

- Hypnos 的实现细节来自本仓库代码，文中以 `文件:行号` 标注。
- 论文层面的数字大多来自检索引擎对原文的摘要。本会话环境的网络策略拒绝直接访问 arxiv.org 与 huggingface.co，无法逐页核对原文。写入论文前，请逐条回原文确认。
- 标注"实测"的数字由本仓库代码复现，脚本为 [`amplitude_probe.py`](amplitude_probe.py)（只依赖 numpy 与 scipy）。
- "创新性"判断基于 2026 年 9 月的检索结果，可能遗漏同期工作。投稿前建议针对选定方向再做一次定向检索。

**术语**

- **per-modality tokenizer**：每类信号（EEG、EOG、EMG、ECG、呼吸）有独立的 SEANet + RVQ tokenizer，把波形变成离散码（`src/hypnos/models/tokenizer/`）。
- **per-second motif**：每个模态每 1 s 窗口得到的离散码（Hypnos 中为 K 级 RVQ 码字），作为生理"语言"的基本词单元。TFM-Tokenizer 把类似的离散单元称为时频 motif [9]。

---

## 0. 结论摘要

1. **推荐主线**（详见第 5 节）：以"语义化 + 带尺度的 per-second motif"为底层（A1 + A2），在其上组合"多秒生理词"（A4 路线 1）与"整夜层级生成"（B2），训练目标加入多尺度未来预测（B3）。评测除常规下游任务外，增加基于似然的零样本检测与"生理语法"最小对评测（D1）。暂定叙事为 *From motifs to nights*：从每秒 motif 到多秒生理词，再到整夜结构的层级生成式语言模型。
2. **创新性与可行性综合最高的方案**：A1（语义蒸馏 motif）、B2（全夜层级生成）、B6（注释流与流式评分）、C1（Rho-1 式 token 选择）、D1（惊异度零样本检测与 PhysioBLIMP）。
3. **不需要重新训练、可立即出结果的实验**：D1 首版（用已发布权重计算逐秒似然）、B7(a)（逐层线性探针）、A2 的幅度诊断（本报告已实测）。
4. **与最强竞品的差异**：SleepFM（留一模态对比）、OSF（比较四类目标后采用 DINO 式自蒸馏 + 通道掩码）、Stanford Sleep Bench 评测的方法都属于编码器式表征学习；SleepMaMi 建模整夜但不是自回归生成式；Physiology as Language 是呼吸到 EEG 的两两翻译。"离散 motif + 层级自回归生成 + 基于似然的零样本能力"这一组合在检索中未见。
5. **主要风险**：(a) 审稿人认为是"把 LLM 与语音技巧搬过来"，需要每个组件都有生理动机和同算力消融；(b) Stanford Sleep Bench 显示对比学习在疾病与死亡预测上显著优于其他预训练方式 [5]，纯生成目标在这类任务上可能落后，需要预置 B8；(c) 本仓库只含推理代码与模型定义，训练循环、数据管线与 tokenizer 的对抗损失需要自行实现。

---

## 1. Hypnos 现状分析

### 1.1 技术谱系：组件、借鉴来源与代码位置

| 组件 | Hypnos 的实现 | 借鉴来源 | 代码位置 |
|---|---|---|---|
| 预处理 | 因果 IIR 滤波 + EMA 滚动 z-score（τ = 60 s）+ 对数幅度压缩 | 流式信号处理惯例 | `src/hypnos/data/preprocessing.py:264`、`:362` |
| 编码器/解码器 | SEANet 卷积 + 可选 RoPE Transformer，严格因果 | SoundStream [32]、EnCodec [31]、Mimi [26]、BrainOmni [13] | `tokenizer/seanet.py`、`tokenizer/tokenizer.py` |
| 量化 | RVQ：EMA 码本、k-means 初始化、死码替换、量化 dropout、可选 rotation trick | Mimi 的 core_vq [26]；rotation trick [44] | `tokenizer/quantizer.py` |
| 连续模式 | VAE 瓶颈（代码存在，发布模型使用离散模式） | VAE | `tokenizer/tokenizer.py:186` |
| 序列模型 | 时间 Transformer + 深度 Transformer（每秒逐级生成 K 个码字） | RQ-Transformer [45]、Moshi [26] | `rq_transformer/model.py`、`rq_transformer/depth.py` |
| 多模态交互 | 轴向注意：模态内因果时间注意 + 同一时刻模态间双向注意 | 轴向注意 | `rq_transformer/multimodal_temporal.py:634` |
| 模态子集鲁棒性 | 训练时按中国餐馆过程（CRP）随机分组，限制跨模态注意 | 非参数贝叶斯中的 CRP 随机划分 | `rq_transformer/multimodal_temporal.py:42` |
| 注意力细节 | RoPE、QK-norm、SwiGLU、LayerScale、局部滑窗 + 周期性全局窗（flex_attention）、XSA | LLaMA/PaLM、ViT-22B、Mimi、局部-全局交替结构、XSA (arXiv:2603.09078) | `src/hypnos/models/attention.py` |
| 权重共享 | 按 (signal_type, K, V) 分组共享嵌入与输出头，可选 tied embedding | LLM 的权重共享 | `rq_transformer/model.py:307` 起 |
| 下游使用 | 每模态 1 Hz 嵌入，模态平均 + 30 s 平均池化 + 线性探针 | 表征学习常规做法 | `README.md`、`rq_transformer/model.py:630` |

从检索摘要得到的论文要点 [1]：8 个模态、20,000 余份整夜 PSG；训练时跨模态注意限制在随机子组内；睡眠分期用 1/100 的标注量达到强监督基线水平；ECG 迁移任务上，CinC 2017 与 xECG 持平，Apnea-ECG 与 CPSC 2021 分别高 4% 与 5%；上下文最长测试到 4096 token（约 68 分钟），分期与觉醒检测在 1024-2048 token 饱和，OSA 分类随上下文增长持续提升。作者把小时到天尺度的规律（昼夜相位、多夜规律性、罕见事件聚集）以及传感器位置编码列为未来方向。

### 1.2 从代码与论文归纳的可改进点

| 编号 | 局限 | 证据 |
|---|---|---|
| L1 | motif 偏"声学"：tokenizer 只用波形重建 + commitment 训练，没有语义约束 | `tokenizer/tokenizer.py:296` 起，`forward` 只返回重建与 VQ 损失 |
| L2 | 幅度失明：滚动 z-score 去掉绝对幅度，`mu_track`、`std_track` 被丢弃 | `embedding/pipeline.py:78`；实测见 A2 |
| L3 | 所有模态强制共享 1 s token，与呼吸周期、心动周期不对齐 | `embedding/manifest.py:92-96` |
| L4 | 时间上下文有限：训练用 64-token 滑窗 + 256-token 全局窗；论文最长 4096 token | `embedding/infer.py:94`；论文摘要 [1] |
| L5 | 同一秒内不同模态的码字在给定历史时条件独立（depth 逐模态调用） | `rq_transformer/model.py:565` 起 |
| L6 | 所有位置、所有 RVQ 级等权交叉熵；已有 `token_mask` 接口但未用于数据选择 | `rq_transformer/model.py:37`、`:46`、`:94-99` |
| L7 | 固定 8 个通道，缺 SpO2、气流、PPG、腿动 EMG | `src/hypnos/settings.py:22`；`rq_transformer/model.py:332` |
| L8 | 单向因果，下游只用最后一层 | `rq_transformer/model.py:630`；`README.md` 的池化示例 |
| L9 | 生成与似然能力没有进入评测，只用于合成演示 | `README.md`；`embedding/generate.py` |

### 1.3 竞品定位

| 工作 | 发表 | 预训练目标 | 表示 | 与本计划的关系 |
|---|---|---|---|---|
| Hypnos [1] | arXiv:2606.09605 | 多模态 NTP（RQ-Transformer） | 每模态 RVQ，1 Hz | 起点 |
| SleepFM [2][3] | ICML 2024；Nature Medicine 2026 | 留一模态对比（LOO-CL） | 连续嵌入 | 约 585,000 小时、约 65,000 人；130 种疾病 C-index ≥ 0.75 |
| OSF [4] | ICML 2026 | 比较对比、重建、自回归、自蒸馏四类目标，最终采用 DINO 式自蒸馏 + 通道掩码（随机遮蔽 50% 输入通道） | 连续 | SleepBench：166,500 小时、9 个公开来源；发现现有模型对推理时缺失通道泛化差、通道不变特征学习关键、扩大样本与模型与多来源混合持续提升 |
| Stanford Sleep Bench [5] | arXiv:2512.09591 | 评测 MAE、去噪、对比 | 连续 | 17,467 份记录、163,000 余小时；疾病与死亡预测中对比学习显著领先且收敛更快 |
| SleepMaMi [6] | ICML 2026 | 宏观编码器：人口学引导对比；微观编码器：MAE + 多模态对比 | 双编码器 | 建模整夜结构，但不是自回归生成式；B2 的最近邻 |
| PFTSleep [7] | SLEEP 2025 | 自监督 Transformer，8 小时、7 通道、125 Hz | 连续 | 全夜编码 |
| Physiology as Language [16] | ICML 2026 | 以呼吸为条件、掩码预测 EEG token | EEG 离散词表 | 14 个数据集、28,394 人；B5 的最近邻 |
| TFM-Tokenizer [9] | ICLR 2026 | 单通道 EEG 时频 motif tokenizer | 离散 motif | motif 概念来源之一 |
| CodeBrain [10] | ICLR 2026 | TFDual tokenizer（时/频解耦）+ EEGSSM | 离散 | A1 的相关工作 |
| Sleep2.0 [8] | npj Digital Medicine 2026 | 睡眠 EEG 自监督 | 连续 | 发现 N2 内部微结构携带健康信息，支持 A1、D2 的"阶段内 motif"动机 |
| eeg-gpt [18] | GitHub 项目 | 单通道 EEG NTP | 100 ms VQ token | 更细时间粒度的 NTP 参照 |

---

## 2. 评分标准

- **创新性（1-5）**：5 = 检索未见生理领域先例，且迁移非平凡；4 = 仅在单模态或编码器式生理模型中有类似思想，差异清晰；3 = 已知技术首次用于多模态生成式睡眠模型，差异中等；2 = 增量改进；1 = 纯工程。
- **可行性（1-5）**：5 = 改动局部（单个损失、输出头或模块），小规模验证预计两周内完成，算力低；4 = 中等改动，小规模重训即可验证；3 = 需要较大工程量或较多算力；2 = 高风险；1 = 路径不清。
- **共用基础设施不计入单个方案**：凡需重新预训练的方案，都依赖同一套训练基础设施（NSRR 数据管线、训练循环、tokenizer 的对抗损失与判别器）。本仓库只含推理代码与模型定义，这部分一次性成本在 5.4 节阶段 0 中统一估算，不重复计入各方案的可行性评分。
- **与两大前提的关系**：保持 / 扩展 / 修改。
- **优先级**：综合创新性、可行性、对论文叙事的支撑，以及与竞品的差异。

---

## 3. 方案总览

| # | 方案 | 主要借鉴 | 针对局限 | 与两大前提 | 创新 | 可行 | 优先级 |
|---|---|---|---|---|---|---|---|
| A1 | 语义蒸馏的 split-RVQ motif tokenizer | Mimi、SpeechTokenizer、X-Codec、LaBraM | L1 | 保持 | 4 | 5 | 高 |
| A2 | 尺度侧信道 token | EnCodec 的 scale、RevIN | L2 | 保持 | 3 | 5 | 高 |
| A3 | 生理同步的多速率 motif + 绝对时间位置编码 | HeartLang、SyllableLM、Qwen2.5-VL、Moirai | L3 | 修改（ECG、呼吸） | 4 | 3 | 中 |
| A4 | motif 词汇化：BPE 生理词 / 熵自适应分块 | Acoustic BPE、时间序列 BPE、BLT、H-Net | L3、L4 | 扩展 | 4 | 4 / 2 | 中高 |
| A5 | 传感器无关 tokenizer + 模态扩展 | BrainOmni、REVE、ImageBind | L7 | 扩展 | 2 | 4 | 中 |
| B1 | 跨模态联合 depth（同一秒的多模态"和弦"） | Moshi 多流 depth、MusicGen 码本交错 | L5 | 扩展 | 3 | 4 | 中高 |
| B2 | 全夜层级生成：秒 → epoch → 整夜 | MEGABYTE、VAR、Samba、CALM | L4 | 扩展 | 4 | 3-4 | 高 |
| B3 | 多时间尺度未来预测（MTP + 潜空间预测） | Meta MTP、DeepSeek-V3、LLM-JEPA | L4 | 保持 | 3 | 5 | 高 |
| B4 | 模态专属参数（Mixture-of-Transformers） | MoT、Time-MoE | 容量竞争 | 扩展 | 3 | 5 | 中 |
| B5 | Any-to-any 跨模态生成与补全 | 4M、UL2、FIM | L7、L9 | 保持 | 3 | 4 | 中 |
| B6 | 注释流作为"内心独白"：流式评分与条件生成 | Moshi inner monologue、DSM | L9 | 扩展 | 4 | 4 | 高 |
| B7 | 双向表示与读出层选择 | BST、AIM、Layer by Layer | L8 | 保持 | 2 | 5 | 中（作消融） |
| B8 | 生成 + 对比联合目标 | CoCa、SleepFM LOO-CL | 疾病预测 | 保持 | 3 | 5 | 高（备选） |
| C1 | Rho-1 式信息量感知 token 选择 | Rho-1 | L6 | 保持 | 4 | 4 | 高 |
| C2 | 跨队列配比（DoReMi）+ 元数据条件化（MeCo） | DoReMi、MeCo | 数据异质性 | 保持 | 3 | 4 | 中 |
| D1 | 惊异度零样本事件检测 + PhysioBLIMP 评测 | 似然比 OOD、ZeroSpeech | L9 | 保持 | 4 | 5 | 高 |
| D2 | SAE "motif 词典"与生成式因果验证 | Monosemanticity、InterPLM、Evo 2 | 可解释性 | 保持 | 3 | 5 | 中高 |

第 6.1 节另列 5 个评估后未进入主表的候选（连续 motif + 扩散头、缩放律研究、检索增强与个体化、FSQ 单码本、μP）。

---

## 4. 方案详述

### A1. 语义蒸馏的 split-RVQ motif tokenizer

**借鉴来源**：Mimi（Moshi）把 WavLM 语义蒸馏进第一级码本，并采用"语义 VQ + 并行声学 RVQ"的 split 结构 [26]；SpeechTokenizer 把 HuBERT 蒸馏进 RVQ 第一级 [27]；X-Codec 在 RVQ 前注入语义特征并加语义重建损失 [28]；LaBraM 的 VQ 目标是傅里叶幅值与相位而非原始波形 [11]；XY-Tokenizer 讨论了低码率下的语义与声学冲突 [30]。

**Hypnos 现状**：tokenizer 只用波形重建与 commitment 训练（`tokenizer/tokenizer.py:296` 起）。RVQ 第一级量化的是方差最大的波形成分，未必对应睡眠生理状态。RQ-Transformer 对全部 K 级码字等权建模。

**方案**

- 每个模态的 tokenizer 增加一个并行的"语义 VQ"（Mimi 式 split：语义 VQ + 并行的 K-1 级声学 RVQ）。语义 VQ 的输出与教师特征做余弦蒸馏，或以教师聚类 ID 为伪标签（SpeechTokenizer 的两种做法）。
- 教师三选一，作为消融：
  - T1 可解释生理描述子（逐秒）：EEG/EOG 的 δ/θ/α/σ/β 相对功率，纺锤波、慢波的检测概率（可用 YASA [25]），EMG RMS，ECG 的 RR 间期与 HRV，呼吸率与呼吸努力幅度。
  - T2 开源对比式 PSG 编码器的特征（SleepFM [3]、OSF [4] 已公开权重），把 Stanford Sleep Bench [5] 中在疾病预测上占优的对比语义蒸进离散 motif。
  - T3 HuBERT 式迭代 [39]：用 Hypnos 自身 temporal context 特征做 k-means，得到伪标签后重训 tokenizer。
- 语义层可以换成 FSQ 大码本 [72]（X-Codec2 使用 65,536 码、码本利用率约 99% [29]），得到"每模态每秒一个词"的 motif 词表，为 A4 与 D2 提供干净的离散单元。

**与两大前提**：保持。per-modality tokenizer 与 1 s 粒度不变，改变的是 motif 的含义：从"波形形状"变为"生理状态 + 形状"。

**创新性 4/5**：EEG 领域已有时频 motif [9]、时/频解耦 token [10]、文本对齐 tokenizer [12]。本次检索未见在多模态生成式睡眠模型中把教师语义蒸馏进 RVQ 第一级的工作，也未见对"生理描述子教师"与"对比编码器教师"的系统比较。

**可行性 5/5**：只在 tokenizer 训练中增加一项损失，教师特征可离线预计算。需要补齐 tokenizer 训练代码（本仓库不含判别器与训练循环，可参考 EnCodec [31]、Mimi [26] 的开源实现）。

**可行性证据**

- Moshi 的 Mimi 消融（原文表 3）：蒸馏 WavLM 显著改善语义 token 的音素可分性（ABX）；split RVQ 使 MUSHRA 从 57.8 升到 64.0，ABX 从 6.5% 小幅变差到 8.1%。说明语义蒸馏与重建存在冲突，而 split 结构可以改善二者的折中 [26]。
- X-Codec 引入语义信息后，TTS 的 WER 显著下降，且收益延伸到音乐续写与文本生成音效 [28]。
- SpeechTokenizer 的第一级蒸馏 HuBERT 后，自回归模型只需建模第一级即可捕获内容信息 [27]。
- 生理侧：TFM-Tokenizer 在 4 个 EEG 基准上 Cohen's κ 最多提升 11%，ear-EEG 睡眠分期提升 14%，部分码字与 K 复合波、纺锤波对应 [9]；LaBraM 以频谱为 VQ 目标（ICLR 2024 spotlight）[11]。

**最小验证实验**：在约 2,000 夜的子集上重训 EEG 与 ECG tokenizer（无蒸馏、T1、T2、T3 四组）。指标：只用语义层 token 的线性探针分期 κ；token 与睡眠分期的 NMI 与 purity；重建 SNR 与频谱误差；画"重建-语义"帕累托曲线。再训练 50M 至 100M 参数的 RQ-Transformer 比较下游任务。

**风险**：语义蒸馏损害重建（Mimi 与 XY-Tokenizer 已观察到）；教师偏差被固化进词表（T1 可解释，但可能遗漏未知生理模式）。

**参考文献**：[3][4][5][9][10][11][12][25][26][27][28][29][30][39][72]

---

### A2. 尺度侧信道 token：恢复被滚动归一化抹掉的绝对幅度

**借鉴来源**：EnCodec 的 48 kHz 模型按 1 s 分块归一化，`encode` 对每块输出 `(codes, scale)`，解码时用 scale 还原幅度 [31]；RevIN 在归一化后保留统计量并在输出端还原 [78]；Chronos 对序列做均值缩放后再量化 [80]。

**Hypnos 现状**：`causal_preprocess_signal`（`preprocessing.py:362`）= 因果滤波 + EMA 滚动 z-score（τ = 60 s，`rolling_normalize` 在第 264 行）+ 对数压缩。`pipeline.py:78` 取回的 `mu_track`、`std_track` 被丢弃，tokenizer 与 RQ-Transformer 都看不到尺度。

**实测**（用本仓库的预处理函数，脚本见 [`amplitude_probe.py`](amplitude_probe.py)，输出见 6.2 节）

- 全局增益完全不可见：`pre(x)` 与 `pre(0.3x)` 的最大差为 0.0。
- 类低通气事件（0.25 Hz 呼吸样信号，幅度下降 30%）：事件持续 10 s 时预处理后仍可见约 25.6% 的下降，30 s 时约 22.2%，60 s 时约 18.2%，120 s 时约 9.5%。下降 50% 的事件在 120 s 时只剩约 21.2%。
- 6 小时内慢波幅度线性下降（原始信号首末小时包络比 0.57），预处理后比值为 1.15，整夜的下降趋势被抹去。

**临床相关性**：AASM 规则中，N3 的判读依赖 0.5-2 Hz、峰峰值大于 75 µV 的慢波；低通气（推荐规则 1A）要求信号幅度相对事件前基线下降至少 30%、持续至少 10 s，并伴随至少 3% 的血氧下降或觉醒 [23]。慢波活动是睡眠稳态过程 S 的指标，在夜间逐渐下降，并随年龄降低 [24]。当前输入对这些量不可见或只部分可见。

**方案**

- 每个模态每秒增加一个"尺度 token"：对 log σ_t（可选加上 μ_t，以及按 EDF 物理单位校准的绝对幅度）做 64-256 档量化（或 FSQ）。输入端与该模态的 RVQ 嵌入相加（`model.py` 的 `_aggregate_embeddings`），输出端作为 depth transformer 的额外一级：先预测尺度，再预测形状。
- 设备与单位差异：用录音内的稳健参考（如整夜 NREM 的中位幅度）得到相对尺度，另附每夜一次的绝对尺度 token；同时用 C2 的元数据条件化，避免模型学到设备捷径。

**与两大前提**：保持。尺度 token 是 per-second motif 的"音量"补充。

**创新性 3/5**：技术本身简单（EnCodec、RevIN 已有），贡献在于指出并量化 token 化睡眠模型的幅度盲区，给出与 AASM 规则对齐的修复。审稿人可能视为工程改进，需要用低通气、N3、年龄任务上的显著提升来支撑。

**可行性 5/5**：尺度序列在预处理中已经算出；tokenizer 无需重训；只需修改 RQ-Transformer 的嵌入层与 depth 输出层。

**可行性证据**：上面的实测；EnCodec 代码中 `encode` 返回 `(codes, scale)` [31]；RevIN 已被大量时间序列预测模型采用 [78]。

**最小验证实验**：同一配置下训练加与不加尺度流的小模型，比较 AHI 回归、低通气检测、N3 召回率、年龄回归；再做增益扰动测试，确认新模型能区分不同幅度。

**风险**：尺度信息可能引入设备或站点捷径（放大器、电极阻抗差异），需要跨队列评估与站点条件化。

**参考文献**：[23][24][31][78][80]

---

### A3. 生理同步的多速率 motif 与绝对时间位置编码

**借鉴来源**：HeartLang 把心搏当作词、节律当作句，用 QRS-Tokenizer 生成"ECG 句子" [15]；SyllableLM 学到可控速率的音节级语义单元（最低 5 Hz、60 bps）[34]；Qwen2.5-VL 把 MRoPE 的时间位置 id 与绝对时间对齐，支持小时级视频与秒级事件定位 [74]；Moirai 按采样频率使用不同的 patch 投影层 [79]。

**Hypnos 现状**：`manifest.py:92-96` 强制所有模态共享 `token_duration_sec`（1 s）。呼吸周期约 3-5 s，1 s token 只覆盖一个呼吸周期的一部分；心搏与 1 s 边界不对齐；纺锤波（0.5-2 s）与 K 复合波会跨越 token 边界。

**方案**

- EEG/EOG/EMG 保留 1 s motif。ECG 改为心搏同步 token（R 峰分段，每个 token = 一个心动周期的形态码 + RR 间期尺度码）。呼吸改为呼吸周期同步 token，或 4 s 固定窗。SpO2 每 5 s 一个 token。检测失败的片段回退到固定窗。
- temporal transformer 的 RoPE 位置改用 token 中心的真实时间（秒，连续值）；modality attention 改为按时间窗的跨模态注意（token 只看 |Δt| < δ 的其他模态 token），可用 flex_attention 的 mask 函数实现（本仓库 `models/attention.py` 已基于 flex_attention）。
- 多速率流的两种布局作为消融：按时间排序交错；按时间窗分组的轴向注意。

**与两大前提**：保持 per-modality tokenizer；把 per-second motif 推广为"按生理周期划分的 motif"（EEG 仍为每秒）。这是对前提 2 的有意修改，论文中需要论证"motif 的单位应由生理节律决定"，并保留 1 s 版本作为对照。

**创新性 4/5**：心搏为词在 ECG 单模态中已有 [15]。本次检索未见在多模态、多速率的生成式睡眠模型中统一时间轴的做法。

**可行性 3/5**：ECG 质量好时 R 峰检测成熟，但呼吸周期在体位变化和低通气期间较难检测；变长序列的批处理与缓存复杂；CRP 分组与 modality attention 需要重写。

**可行性证据**：HeartLang 在 6 个公开 ECG 数据集上与其他自监督方法相比有竞争力 [15]；SyllableLM 报告训练算力降低 30 倍、推理提速约 4 倍，并匹配或超过当时的 SpeechLM [34]；Qwen2.5-VL 报告绝对时间编码支持小时级视频理解与秒级定位 [74]。

**最小验证实验**：只把 ECG 改为心搏 token，其他模态不变；在 Hypnos 已报告的 ECG 任务（CinC 2017、Apnea-ECG、CPSC 2021）与夜间 HRV 相关的疾病预测上对比。

**风险**：检测误差传播；心动过速时序列变长；与 A4 的动态分块功能部分重叠，二者可择一。

**参考文献**：[15][34][74][79]

---

### A4. motif 词汇化：从 per-second motif 到多秒"生理词"

**借鉴来源**：文本 BPE；Acoustic BPE 把高频离散语音单元组合合并为新 token [33]；时间序列 BPE 学习高频 motif 词表并自适应压缩 [63]；BLT 按下一字节的熵动态切分 patch [51]；H-Net 端到端学习分块 [52]；SyllableLM [34]。

**Hypnos 现状**：固定每秒一个 token。数分钟的稳定 N3 或安静清醒，与觉醒、呼吸事件等事件期消耗同样多的 token；上下文受限于训练窗口（L4）。

**方案**（两条路线，风险递增）

- **路线 1（BPE，低风险）**：在 A1 语义层的 motif 序列上按模态训练 BPE，得到多秒"生理词"，例如一个完整呼吸周期、K 复合波后接纺锤波、呼吸暂停后的恢复性呼吸。全局模型在"词"序列上做 NTP，局部模型在每秒内生成声学残差码，对应 SpeechTokenizer 与 VALL-E 中"自回归建模语义层 + 非自回归生成声学层"的分工 [27][43]。词表本身作为可解释产出：统计是否符合 Zipf 分布，按睡眠分期与疾病比较词频。
- **路线 2（动态分块，高风险）**：用小型 NTP 模型的逐秒熵决定分块边界（BLT），或采用 H-Net 式可微分块。低熵的平稳段合并，高熵的事件段保持 1 s 粒度。

**与两大前提**：扩展。per-second motif 仍是最小单位，"词"建立在 motif 之上。多模态对齐需要时间戳，可与 A3 的绝对时间位置编码配合。

**创新性 4/5**（路线 1 约 3，路线 2 约 5）：本次检索未见在睡眠或多模态生理模型中学习多秒离散词表或动态分块的工作；时间序列 BPE 已有单变量预测工作 [63]。

**可行性 4/5（路线 1）/ 2/5（路线 2）**：在离散 ID 上训练 BPE 的开销很小；难点在于多模态合并不同步后的对齐，以及与 depth 生成的接口。

**可行性证据**

- Acoustic BPE 在 LibriTTS 上提高了合成语音的可懂度与多样性，并加快推理 [33]。
- 时间序列 BPE 的原文摘要称：在多个时序基础模型上预测性能平均提升约 40%，效率提升显著（原文给出 2314%，需核对口径）[63]。
- BLT 在 FLOP 对齐条件下与 Llama 3 持平，推理 FLOPs 最多减少 50%，缩放实验达 8B 参数、4T 字节 [51]。
- H-Net 在 DNA 等缺乏好分词启发式的模态上数据效率提升接近 4 倍 [52]。

**最小验证实验**：在语义层 token 上训练 BPE，报告整夜 token 数的压缩倍数、词与 AASM 事件的对应关系，以及同等算力下因上下文变长带来的 OSA 与疾病预测变化。

**风险**：词分布在不同队列间漂移；合并可能模糊事件边界的精确时间，事件检测需要局部模型补偿。

**参考文献**：[27][33][34][43][51][52][63]

---

### A5. 传感器无关的 per-modality tokenizer 与模态扩展

**借鉴来源**：BrainOmni 的 Sensor Encoder 编码电极的三维坐标、朝向与类型，再与 SEANet 特征融合后做 RVQ [13]；REVE 用 4D 位置编码支持任意电极布局与长度 [14]；ImageBind 通过配对数据把新模态绑定到已有嵌入空间 [75]；OSF 发现现有睡眠模型对推理时缺失通道泛化差，通道不变特征学习对预训练关键 [4]；Hypnos 作者也把传感器位置编码列为未来方向 [1]。

**Hypnos 现状**：`settings.py:22` 固定 8 个通道（C3、C4、E1、E2、Chin、ECG、ABD、THX），通道嵌入表只有 8 行（`model.py:332`）。缺 SpO2、气流（鼻压、热敏）、PPG、腿动 EMG、体位、鼾声。AASM 呼吸事件评分依赖气流与血氧 [23]，当前模型只能从胸腹带间接推断。

**方案**

- 按模态族建 tokenizer（EEG、EOG、EMG、ECG、呼吸努力、气流、SpO2、PPG、体动），加入传感器元数据条件（电极坐标或导联、设备类型、采样率）。EEG 族支持 F3/F4/O1/O2、耳道 EEG、额部可穿戴 EEG。
- 模态扩展协议：冻结主干，新模态的 tokenizer 与适配器先在配对数据上对齐（ImageBind 式），再联合微调，并评估旧任务上的遗忘。

**与两大前提**：扩展。per-modality tokenizer 从"每通道"推广到"每模态族 + 传感器条件"；motif 仍为 1 s。

**创新性 2/5**：传感器编码在 EEG 基础模型中已有，增量在于扩展到多生理信号的生成式模型。临床价值高，但单独作为顶会贡献偏弱。

**可行性 4/5**：NSRR 多数队列有 SpO2 与气流；各队列的导联命名差异可以复用 `data/edf.py` 的别名机制处理。

**可行性证据**：REVE 以 92 个数据集、60,000 余小时 EEG 预训练，在 10 个下游任务上达到 SOTA [14]；BrainOmni 在 EEG 与 MEG 的异构设备间泛化 [13]；OSF 的三条发现，以及其通道掩码（随机遮蔽 50% 输入通道）对缺失通道鲁棒性的改善 [4]。

**最小验证实验**：加入 SpO2 与鼻压两个模态，评估 AHI 估计误差与低通气检测是否改善（建议与 A2 联合做）。

**风险**：数据异构与缺失率上升，需要与 CRP 分组配合。

**参考文献**：[1][4][13][14][23][75]

---

### B1. 跨模态联合 depth：把同一秒的多模态 motif 当作"和弦"建模

**借鉴来源**：Moshi 的 Depth Transformer 在每个时间步联合生成文本 token 与多路音频流的多级码本，并在语义码与声学码之间引入 acoustic delay [26]；MusicGen 系统比较了 flattening（精确分解）、delay、parallel 等码本交错方式，delay 以约 1/4 的计算量达到与 flattening 接近的质量 [35]。

**Hypnos 现状**：`model.py:565` 起对每个模态单独调用 depth transformer。给定历史时，同一秒内不同模态的码字条件独立：第 t 秒 EEG 的码字看不到第 t 秒 ECG 的码字。生理上同一秒内存在强耦合，例如 C3 与 C4 高度相关、眼动同时出现在 EOG 与额区 EEG、ECG 伪迹进入 EEG、觉醒时 EEG 与 EMG 同步变化。当前这些关系只能经由"上一秒"间接建模。

**方案**

- 联合 depth 序列按"各模态语义 motif（A1）→ 各模态声学残差"排列，即语义优先、跨模态、再声学；或按 MusicGen 的 delay 模式把声学级延迟一步，缩短 depth 长度。
- CRP 分组同样作用于联合 depth 的注意 mask，保持模态子集鲁棒性。

**与两大前提**：扩展。per-second motif 扩展为"每秒多模态和弦"。

**创新性 3/5**：多流 depth 在语音中已有，生理多模态中未见。生成保真度的收益较确定，表征收益不确定。

**可行性 4/5**：`SharedDepthTransformer` 已共享层与位置嵌入（`depth.py`），改为变长联合序列即可。depth 长度从 K_m 增至 ΣK_m（8 个模态 × 4-8 级 = 32-64），计算可控。

**可行性证据**：Moshi 的多流 depth 与 acoustic delay [26]；MusicGen 的交错模式对比 [35]；Hypnos 的 `synthesize` 提供了直接的生成质量评测接口（`embedding/generate.py`）。

**最小验证实验**：比较独立 depth 与联合 depth 的跨模态一致性指标（C3/C4 相干性、ECG 伪迹在 EEG 中的相位关系、觉醒时 EMG 与 EEG 的同步）、各模态 NLL 与下游表征。

**风险**：depth 变长使训练变慢；若 temporal context 已包含足够信息，表征收益可能很小，需如实报告。

**参考文献**：[26][35]

---

### B2. 全夜层级生成：秒级 motif → epoch 级潜变量 → 整夜

**借鉴来源**：MEGABYTE 的全局/局部两级解码器 [53]；VAR 的 next-scale 预测（NeurIPS 2024 最佳论文）[69]；Samba 的 Mamba + 滑窗注意混合结构 [54]；CALM 把 K 个 token 压成一个连续向量做 next-vector 预测 [55]。睡眠侧最接近的是 SleepMaMi 的宏观/微观双编码器 [6]，但其目标为对比与 MAE，不能自回归采样整夜。

**Hypnos 现状**：temporal transformer 训练时为 64-token 滑窗 + 256-token 全局窗（`infer.py:94`），论文中上下文最长 4096 token（约 68 分钟）。论文报告 OSA 分类随上下文增长持续提升，作者把小时到天尺度的规律列为未来方向 [1]。一个睡眠周期约 90 分钟，整夜约 28,800 s，超出当前可见范围。

**方案**

- 局部层：保留 Hypnos 式每秒多模态 NTP，窗口 30-120 s。
- 全局层：每 30 s epoch 用学习到的池化或 CALM 式自编码器得到 epoch 潜变量（一夜约 960 个），全局因果 transformer 在其上做 next-epoch 预测（连续潜变量用能量分数或扩散头，或离散化为 epoch 级 motif），并以 MEGABYTE 的方式作为条件注入局部层。
- 可选：VAR 式由粗到细的 next-scale 生成（整夜 → 小时 → epoch → 秒）；或 Samba 式混合骨干直接在 1 Hz 序列上建模整夜。

**与两大前提**：扩展。per-second motif 是最底层，新增更高层级。

**创新性 4/5**：睡眠中已有全夜建模 [6][7]，但"生成式、跨尺度、可流式"的层级 NTP 未见。它支持整夜睡眠结构轨迹的采样与生成式评估。

**可行性 3/5（需额外训练 epoch 级 tokenizer 时）至 4/5（端到端学习池化时）**：全局序列约 960 个位置，计算不是瓶颈。主要工作是按整夜组织数据加载。规模估算：2 万夜 × 8 h × 3600 s × 8 个模态 ≈ 4.6 × 10^9 个"模态-秒"位置（每个位置含 K 个码字）。

**可行性证据**：MEGABYTE 在 arXiv 数据上 0.678 bpb，优于 Transformer 的 0.816 与 PerceiverAR 的 0.791 [53]；Samba 在 4K 长度上训练，零样本在最长 1M 上下文上困惑度改善，微调后外推到 256K 并完美完成 passkey 检索 [54]；VAR 呈现与 LLM 类似的幂律缩放 [69]；Hypnos 自身的 OSA 结果随上下文增长持续改善 [1]。

**最小验证实验**：固定局部模型，增加全局层；在 OSA 严重度、疾病与死亡预测（C-index）、睡眠结构指标（REM 潜伏期、睡眠周期数）上与 Hypnos 的 4096-token 版本比较。

**风险**：全局层可能只学到年龄、性别等人口学信息的捷径，需要做混杂控制分析。

**参考文献**：[1][6][7][53][54][55][69]

---

### B3. 多时间尺度的未来预测目标（MTP + 潜空间预测）

**借鉴来源**：Multi-token prediction [46]、DeepSeek-V3 的 MTP 模块 [47]；LLM-JEPA 在 LLM 训练中加入嵌入空间预测 [56]；EEG 侧，Banville 等的时间上下文预测与 CPC [22]。

**Hypnos 现状**：只预测下一秒。平滑生理信号的下一秒预测可以大量依赖局部延续，睡眠分期转换、周期等缓慢动态只在较远的预测中才有强梯度。这是一个假设，目前没有直接证据，需要通过下面的最小验证实验检验。

**方案**

- 额外的预测头：预测 t+5 s、t+30 s、t+300 s 的语义 motif（A1），或预测"未来 30 s 的 motif 词袋分布"（多标签交叉熵）。后者代价低，且对慢动态敏感。
- JEPA 项：用 EMA 目标编码器得到未来 epoch 的表征，当前状态经预测器回归（余弦或平滑 L1 损失）。

**与两大前提**：保持。

**创新性 3/5**：MTP 与 JEPA 是已知技术；在生理生成模型中做多尺度预测，并系统分析其对慢任务的收益，检索未见。

**可行性 5/5**：只增加若干输出头，可复用 `_compute_per_level_loss`。

**可行性证据**：13B 参数的 MTP 模型在 HumanEval 上多解出 12% 的题目、MBPP 上多 17%，且有利于 induction head 的形成，4-token 预测的推理最多快 3 倍 [46]；LLM-JEPA 在 Llama3、OpenELM、Gemma2、OLMo 等模型族上优于标准训练目标，并更抗过拟合 [56]；Banville 等报告，基于时间上下文的自监督特征在低标注条件下优于纯监督模型 [22]。

**最小验证实验**：3 个预测跨度 × 2 种目标（token 与词袋），比较分期、OSA、年龄、疾病任务。

**风险**：远期预测的不确定性高，模型可能主要学到先验，因此建议用词袋分布等可预测的统计量作为目标。

**参考文献**：[22][46][47][56]

---

### B4. 模态专属参数（Mixture-of-Transformers）

**借鉴来源**：Mixture-of-Transformers（MoT）按模态解耦 FFN、注意力投影与 LayerNorm，保留全局自注意 [57]；时间序列侧的 Time-MoE [81]。

**Hypnos 现状**：temporal 层对所有模态共享权重（`multimodal_temporal.py` 中输入被 reshape 为 (B·M, S, D) 送入同一层），只有嵌入与输出头按信号类型分组。

**方案**：在 temporal 层中按信号族（EEG/EOG、EMG、ECG、呼吸）索引 FFN 与 QKV 权重；modality attention 层保持共享，负责跨模态交互。Hypnos 的轴向设计使这一改动几乎不增加 FLOPs。

**与两大前提**：扩展。把 per-modality tokenizer 的思想延伸到 per-modality 变换参数。

**创新性 3/5**；**可行性 5/5**。

**可行性证据**：MoT 在 Chameleon 7B 设置下以 55.8% 的 FLOPs 匹配稠密基线；加入语音作为第三模态时，语音部分只需 37.2% 的训练 FLOPs；在 Transfusion 设置下，760M 的 MoT 优于 1.4B 的稠密模型 [57]。EEG（0.5-45 Hz）与呼吸（约 0.1-0.5 Hz）的频带与动力学差异很大，共享 FFN 存在容量竞争是合理假设。

**最小验证实验**：同 FLOPs 下比较共享权重与 MoT 的各模态 NLL 与下游表现，并分析模态间的梯度冲突。

**风险**：参数量增加后，数据量较小的模态（如 EMG）可能过拟合，可让小模态共享部分参数。

**参考文献**：[57][81]

---

### B5. Any-to-any 跨模态生成与补全（UL2 / 4M 式混合目标）

**借鉴来源**：4M 与 4M-21 为每个模态训练专属 tokenizer，用统一 transformer 做任意子集到任意子集的掩码建模 [68]；UL2 的混合去噪器 [58]；FIM 把中间片段移到末尾即可让因果模型学会补全，且不损害从左到右的能力 [59]；睡眠侧，Physiology as Language 把 EEG 离散化后，以呼吸为条件用掩码预测翻译 EEG [16]。

**Hypnos 现状**：CRP 分组训练让模型对模态子集鲁棒，但目标始终是"历史 → 下一秒"，没有显式训练"其他模态的当前与历史 → 缺失模态的当前"。

**方案**：训练时按比例混合三种任务，用任务嵌入区分：(i) 标准 NTP；(ii) 跨模态翻译：目标模态在 [t, t+Δ] 内被隐藏，条件为源模态直到 t+Δ（DSM 式延迟 [36]）；(iii) 片段补全（FIM），用于伪迹修复。

**与两大前提**：保持。正是 per-modality tokenizer 让 any-to-any 可行，这也是 4M 的核心设计。

**创新性 3/5**：两两翻译已有 [16]；单一模型覆盖所有方向、同时兼顾 NTP 表征学习，是增量创新。

**可行性 4/5**：主要是注意 mask 与数据管线的工程。

**可行性证据**：Physiology as Language 使用 14 个数据集、28,394 人、33,919 夜；由呼吸合成的 EEG 用于分期的准确率为 0.84（真实 EEG 为 0.88），年龄 MAE 5.0 年（真实 5.1 年），性别 AUROC 0.81（真实 0.82）[16]；FIM 在多种规模下不损害从左到右的困惑度 [59]；4M 可以在任意模态组合上做条件生成 [68]。

**最小验证实验**：ECG + 呼吸 → EEG/EOG/EMG token，评估合成信号上的分期准确率与频谱误差，并与 [16] 的数字对照（注意数据集不同）。

**风险**：与 B6、D1 共享大量实现，建议合并实施。

**参考文献**：[16][36][58][59][68]

---

### B6. 注释流作为"内心独白"：可控延迟的流式评分与条件生成

**借鉴来源**：Moshi 的 inner monologue，即与音频时间对齐的文本 token 流 [26]；Kyutai 的 Delayed Streams Modeling（DSM）用同一个解码器建模已对齐的多条流，文本流延迟即为 ASR，音频流延迟即为 TTS [36]；NeuroLM 的指令微调 [12]；SensorLM 的分层文字描述 [21]。

**Hypnos 现状**：专家标注只在下游线性探针中使用。

**方案**

- 把专家标注作为额外的"模态"：分期流（30 s 标签广播到 1 Hz），事件流（觉醒、阻塞性/中枢性呼吸暂停、低通气、血氧下降、周期性腿动的起止 token）。每条流有自己的小词表。
- 有标注的夜晚联合训练 p(信号, 标注)；无标注的夜晚把注释流视为缺失（Hypnos 的 `modality_mask` 已支持缺失模态）。
- 推理：注释流延迟 D 秒 → 带 D 秒前瞻的流式评分；以注释为条件生成信号 → 按给定 AHI 或分期序列合成生理信号，可用于数据增强、算法测试与教学。

**与两大前提**：扩展。注释流拥有自己的 tokenizer（小词表）。

**创新性 4/5**：本次检索未见"生成式联合模型 + 可控延迟的流式评分 + 标注条件生成"的睡眠工作。

**可行性 4/5**：NSRR 多数队列提供 XML 格式的分期与事件标注；实现上是新增两条离散流。

**可行性证据**：DSM 在 ASR 与 TTS 上报告了 SOTA 的性能与延迟，并支持任意长序列 [36]；Moshi 使用 inner monologue 改善生成语音的语言质量 [26]。

**最小验证实验**：D ∈ {0, 5, 15, 30} s 的流式分期 κ 曲线，并与离线分期对比；用独立分类器检验合成事件是否可被识别。

**风险**：审稿人可能认为本质是多任务监督。需要强调半监督与生成能力，并通过去掉注释流的消融证明预训练表征没有因此退化。

**参考文献**：[12][21][26][36]

---

### B7. 双向表示与读出层选择（面向离线任务）

**借鉴来源**：Belief State Transformer 用前向与后向编码器联合预测下一个与上一个 token [60]；FIM [59]；AIM 的 prefix 因果注意，以及"浅层特征优于最后一层"的发现 [71]；Layer by Layer 在 32 个文本嵌入任务上发现中间层一致更强 [61]。

**Hypnos 现状**：因果模型，下游只用最后一层的 temporal context（`model.py:630` 的 `embeddings["1s"]`），30 s 池化后的向量只看过去。

**方案**：(a) 多层读出：逐层线性探针 + 可学习的层加权；(b) 训练反向模型或 BST，离线任务拼接前向与后向状态；(c) AIM 式 prefix 双向注意，使同一模型在下游可开启双向。

**与两大前提**：保持。

**创新性 2/5**；**可行性 5/5**（(a) 可用已发布权重立即完成）。

**可行性证据**：AIM 7B 模型在 20 亿张图像上预训练，冻结主干的 ImageNet-1k 准确率为 84.0%，且更浅层的特征质量更高 [71]；Layer by Layer 的结论 [61]。AASM 判读本身会参考前后 epoch，离线分期天然受益于未来上下文。

**最小验证实验**：用已发布权重在 DOD-O 上做逐层线性探针（`demo.ipynb` 已有完整流程），报告最佳层与提升幅度。

**风险**：收益可能有限，适合作为消融而非主要贡献。

**参考文献**：[59][60][61][71]

---

### B8. 生成 + 对比的联合目标（CoCa 式）

**借鉴来源**：CoCa 同时使用对比损失与生成（captioning）损失 [73]；SleepFM 的留一模态对比 [3]；Stanford Sleep Bench 发现对比学习在死亡与疾病预测上显著优于其他预训练方法，且收敛更快 [5]；OSF 发现通道不变特征学习对预训练至关重要 [4]。

**Hypnos 现状**：纯生成目标。生成目标擅长局部动态，但 [5] 的结论提示，疾病预测可能更受益于对比式的跨模态不变性。

**方案**：在 epoch 级（30 s 或 5 min）对各模态的池化表征加 LOO-CL 或 SigLIP 式成对 sigmoid 损失，与 NTP 联合训练；对比分支只增加轻量投影头。

**与两大前提**：保持。

**创新性 3/5**；**可行性 5/5**。

**可行性证据**：CoCa 的 ImageNet 零样本准确率为 86.3% [73]；SleepFM 在 130 种疾病上 C-index ≥ 0.75 [2]；Stanford Sleep Bench [5] 与 OSF [4] 的结论。

**最小验证实验**：同算力下比较 NTP、对比、NTP + 对比三种目标在分期、OSA、疾病 C-index 上的表现。

**风险**：两种目标的权重需要调节，可能削弱生成质量，需要同时报告 NLL 的变化。

**参考文献**：[2][3][4][5][73]

---

### C1. 信息量感知的 token 选择（Rho-1 式选择性 NTP）

**借鉴来源**：Rho-1 用参考模型的超额损失为 token 打分，只在高分 token 上训练 [48]。

**Hypnos 现状**：所有位置、所有 RVQ 级等权交叉熵（`model.py:37` 起）。平稳段的 token 容易预测，细粒度 RVQ 级多为噪声，电极脱落与体动伪迹产生不可学习的高损失 token。`_compute_per_level_loss` 已支持 `token_mask`（`model.py:46`、`:94-99`），实现改动小。

**方案**：在人工质控的干净子集上训练小参考模型；对全量数据计算每个（模态，秒，RVQ 级）的超额损失（当前模型损失减去参考模型损失）；按分位数选择或连续加权。对照两种简单基线：只按 token 熵、只按信号质量指数（SQI）。

**与两大前提**：保持。

**创新性 4/5**；**可行性 4/5**。

**可行性证据**：Rho-1 在 80B token 的通用预训练中，15 个任务平均提升 6.8%；数学领域以 3% 的预训练 token 达到 DeepSeekMath 的水平；获 NeurIPS 2024 最佳论文亚军 [48]。Hypnos 代码已有 token 级掩码接口。

**最小验证实验**：训练 token 比例取 100%、60%、30%，比较下游曲线与训练效率。

**风险**：被过滤的"低损失"区间可能包含重要但可预测的结构（如稳定 N2），需要按分期分层检查。

**参考文献**：[48]

---

### C2. 跨队列数据配比（DoReMi）与元数据条件化（MeCo）

**借鉴来源**：DoReMi 用 Group DRO 训练的小代理模型学习域权重 [49]；MeCo 在训练前段把 URL 等元数据放在文本前，最后冷却阶段去掉 [50]；OSF 发现多来源数据混合的扩展持续提升下游表现 [4]。

**Hypnos 现状**：多队列数据的配比与站点、设备异质性处理在本仓库中不可见（仓库只含推理部分）。

**方案**：(a) 以队列、年龄段（儿童队列如 CHAT、NCHSDB；成人；老年队列如 MrOS、SOF）、设备为域，用 DoReMi 学配比；(b) MeCo：把队列、设备、采样率、导联作为可学习的元数据前缀嵌入，冷却阶段去掉；推理时可选择性提供，也可用于"站点反事实"分析。

**与两大前提**：保持。

**创新性 3/5**；**可行性 4/5**。

**可行性证据**：DoReMi 用 280M 代理模型为 8B 模型设定域权重，平均少样本下游准确率提高 6.5 个百分点，达到基线准确率所需步数少 2.6 倍 [49]；MeCo 让 1.6B 模型少用 33% 的数据达到同等下游表现，并在 600M 至 8B 规模上一致有效 [50]。

**最小验证实验**：在约 50M 的代理模型上学权重，迁移到主模型；报告留一队列的跨队列泛化。

**风险**：域的划分方式影响结果，需要公开完整配比以便复现。

**参考文献**：[4][49][50]

---

### D1. 基于惊异度的零样本事件检测与 PhysioBLIMP 评测

**借鉴来源**：Ren 等的似然比 OOD 检测，用背景模型校正背景统计量，在基因组生成模型上验证 [82]；Serrà 等指出输入复杂度主导生成模型的似然，需要校正 [83]；ZeroSpeech 2021 的 sWUGGY 与 sBLIMP 通过比较真词/非词、合法/不合法句子的似然评测语言模型 [41]；Cuervo 与 Marxer 发现 SpeechLM 的预训练损失与下游句法、语义表现强相关 [40]。

**Hypnos 现状**：模型具备完整的逐 token 似然（depth transformer + 输出头，前向结果中已有 `per_modality_per_sample_loss`），但评测只用线性探针，生成能力只用于演示。

**方案**

- **零样本检测**：计算逐秒、逐模态的 NLL 时间序列，减去背景模型（如单模态小模型或只用低阶 RVQ 的模型）的 NLL，得到似然比。用于觉醒、呼吸事件、伪迹、异位心搏的零样本检测与排序（AUPRC）。
- **PhysioBLIMP**：构造最小对（真实 vs 扰动），例如跨被试的模态拼接（EEG 来自 A、ECG 来自 B）、心肺解耦（打乱 RR 间期与呼吸相位的对应）、时间反转、分期序列打乱、把事件移植到不合理的位置。以"模型给真实样本更高似然"的比例为指标，衡量模型掌握的"生理语法"。
- **前瞻预警**：采样未来 60 s，估计血氧下降或觉醒的概率。

**与两大前提**：保持，并直接依赖 per-second motif 的离散似然。连续潜变量方案会失去这一能力，这也是 6.1 节不建议把连续 motif 作为主线的原因之一。

**创新性 4/5**；**可行性 5/5**：原始惊异度可用已发布权重在 DOD-O 等公开数据上直接计算，只需把 `_compute_per_level_loss` 改为输出逐位置损失；似然比版本需要额外训练一个小背景模型。

**可行性证据**：似然比方法在基因组 OOD 检测上显著优于原始似然 [82]；ZeroSpeech 指标被 SpeechLM 研究广泛采用 [41]；[40] 的损失与下游能力相关性结论。

**最小验证实验**：在 DOD-O 上做零样本觉醒检测（AUPRC）；PhysioBLIMP 首版包含 5 类最小对，比较 Hypnos 与改进模型。

**风险**：原始似然受信号复杂度影响 [83]，必须报告似然比版本及消融。

**参考文献**：[40][41][82][83]

---

### D2. SAE "motif 词典"：机制可解释与生成式因果验证

**借鉴来源**：Anthropic 的 monosemanticity 系列 [64]；OpenAI 的 TopK SAE，在 GPT-4 激活上训练 1,600 万隐单元 [65]；InterPLM 在 ESM-2 上每层找到最多 2,548 个可解释特征，对应最多 143 个已知生物学概念 [76]；Evo 2 的 SAE 特征对应外显子-内含子边界、转录因子基序、原噬菌体区域 [77]；EEG 侧，arXiv:2605.13930 在 SleepFM、REVE、LaBraM 上用 TopK SAE 与频谱解码器做特征干预 [17]。

**Hypnos 现状**：没有可解释性分析；检索摘要显示作者把"发现"列为长期目标 [1]。

**方案**

- 在 temporal context 与 modality attention 的输出上训练 TopK SAE；用 NSRR 事件标注与 YASA 检测器（纺锤波、慢波）[25] 为特征自动打标签，并统计特征与疾病结局的关联。
- **差异化：生成式因果验证**。放大或抑制某个 SAE 特征，经 depth transformer 采样码字，再用 tokenizer 解码回波形，检验是否生成对应的纺锤波、K 复合波、呼吸暂停等。编码器式基础模型只能借助外部频谱解码器做间接验证 [17]。
- **跨模态特征**：寻找同时在 EEG 流与呼吸流上激活的事件特征，例如"呼吸暂停后的觉醒"。

**与两大前提**：保持。

**创新性 3/5**（SAE 用于 EEG 基础模型已有，生成式验证与跨模态事件特征为增量）；**可行性 5/5**。

**可行性证据**：[76][77] 在蛋白与基因组语言模型上发现大量与已知生物学概念对应的特征；[17] 证明 SAE 可用于睡眠与 EEG 基础模型。

**最小验证实验**：已发布权重 + DOD-O：训练 SAE，报告与 AASM 事件对齐的特征数量与纯度，并给出 3 个特征的生成干预示例。

**风险**：可解释性评价容易主观，需要预先定义自动化指标（与事件标注的对齐 F1、干预效应量）。

**参考文献**：[1][17][25][64][65][76][77]

---

## 5. 推荐组合与论文方案

### 5.1 推荐主线：From motifs to nights

三项核心贡献：

1. **Tokenizer**：A1 + A2（A5 的 SpO2 与气流作为可选扩展）。叙事：Hypnos 的 per-second motif 偏声学且对幅度失明（有实测证据），提出"语义 + 声学 + 尺度"三通道的 per-modality motif tokenizer。
2. **模型**：B2 全夜层级生成 + A4 路线 1（motif BPE）+ B3 多尺度预测；B1 联合 depth 作为生成质量的增强项。
3. **能力与评测**：D1（零样本检测 + PhysioBLIMP）+ 在疾病与死亡预测上与 OSF、SleepFM、SleepMaMi、Hypnos 对比；D2 作为分析章节。

训练策略：C1 作为训练效率改进（可放在消融中）；B8 作为备选辅助损失，若疾病预测落后于对比式模型则启用。

结构示意：

```text
EDF / 多导原始信号
  |
  v
per-modality tokenizer (A1 + A2; A3/A5 可选)
  每个模态每秒: [尺度 s_t] + [语义 motif m_t] + [声学残差 r_t(1..K)]
  |
  v
motif 词汇化 (A4, 可选): m_t 序列 -> 多秒"生理词"
  |
  v
局部层 (30-120 s 窗口): 轴向时间/模态注意, 模态专属参数 (B4, 可选)
  联合 depth: 同一秒的多模态码字按"和弦"生成 (B1)
  |
  v  每 30 s 池化
全局层 (整夜约 960 个 epoch): next-epoch 预测 (B2)
  |
  v
目标: NTP + 多尺度未来预测 (B3) [+ 对比 B8] [+ 注释流 B6]
训练: token 选择 (C1), 配比与元数据条件 (C2)
评测: 线性探针, 零样本惊异度与 PhysioBLIMP (D1), SAE 分析 (D2)
```

选择理由：

- 每个组件都有明确的生理动机：AASM 幅度规则与实测的幅度盲区（A2）、阶段内微结构（A1，参见 [8]）、睡眠周期与整夜结构（B2）、事件形态（A4），不是单纯的技术移植。
- 与最强竞品差异清晰：SleepFM、OSF 与 Stanford Sleep Bench 评测的方法为编码器式；SleepMaMi 的全夜模型不能自回归采样；Physiology as Language 是两两翻译。
- A2 的真实数据诊断与 D1 的原始惊异度评测可以在 2-3 周内出初步结果，降低整体风险。

### 5.2 备选主线

| 主线 | 组合 | 优点 | 缺点 |
|---|---|---|---|
| 应用导向："一个模型完成评分、翻译与模拟" | B6 + B5 + A2 + D1 | 临床价值直观，实现风险低 | 方法新意中等，更适合 CHIL、ML4H 或期刊 |
| 效率与缩放 | C1 + C2 + B4 + 缩放律研究（6.1 节） | 风险低，结论可复用 | 新意有限，审稿人可能认为是工程优化 |

### 5.3 实验设计

- **数据**：NSRR 队列（SHHS、MESA、MrOS、CFS、CHAT、CCSHS、WSC、NCHSDB、SOF 等，需签署数据使用协议）；Stanford Sleep Bench（BDSP，163,000 余小时）[5]；OSF 的 SleepBench（166,500 小时、9 个公开来源）[4]。按队列划分留出集，用于分布外评估。
- **基线**：Hypnos（公开权重，首要对照；最好在相同数据与算力下复现，至少在相同评测协议下比较）、OSF（公开权重）、SleepFM（公开代码）、SleepMaMi、PFTSleep（公开代码）、监督模型（如 U-Sleep）；EEG tokenizer 对照 TFM-Tokenizer、CodeBrain。
- **任务与指标**：分期（κ、macro-F1）；觉醒与呼吸事件检测（事件级 F1、AUPRC）；AHI 回归与 OSA 分级；年龄、BMI 回归；疾病与死亡预测（C-index，Stanford Sleep Bench 的 13 个疾病任务）；缺失模态鲁棒性（沿用 Hypnos 与 OSF 的模态子集评测）；ECG 迁移（CinC 2017、Apnea-ECG、CPSC 2021，与 Hypnos 相同）；生成质量（频谱距离、事件统计、整夜结构指标）；零样本任务（D1）。
- **消融**：每个组件在 50M 至 150M 参数、约 5,000 夜的小规模上单独开关，3 个随机种子；主模型在全量数据上训练一次。
- **算力**：数据规模约 4.6 × 10^9 个"模态-秒"位置（2 万夜的估算，见 B2）。小规模消融预计可在单机 8 卡上以天为单位完成，具体需按最终模型规模核算。

### 5.4 时间线（约 5-6 个月）

目标会议可考虑 ICML 2027 或 NeurIPS 2027，截止日期以官网为准。

| 阶段 | 时长 | 内容 |
|---|---|---|
| 0 | 2-3 周 | 用已发布权重完成 D1 首版、B7(a) 逐层探针、A2 的真实数据诊断；搭建 NSRR 数据管线与 tokenizer 训练代码（本仓库不含训练代码） |
| 1 | 4-6 周 | A1 + A2 tokenizer；小规模语言模型验证 |
| 2 | 6-8 周 | B2 + A4 路线 1 + B3（+ B1），中等规模实验 |
| 3 | 4-6 周 | 全量训练、完整评测、D2 分析、写作 |

### 5.5 审稿风险与应对

| 可能的质疑 | 应对 |
|---|---|
| 只是把 LLM 与语音技巧搬过来 | 每个组件给出生理动机与失败案例（如 A2 的实测），并做同算力消融 |
| 与 Hypnos 的比较不公平 | 同数据、同参数量、同训练 token 数复现 Hypnos；或用其公开权重在相同协议上比较并说明差异 |
| 疾病预测不如对比式模型 | 预置 B8，并在 Stanford Sleep Bench 协议下报告 |
| 生成质量评价主观 | 使用可计算的频谱、事件统计与 PhysioBLIMP 指标 |
| 数据泄漏 | 被试级划分，跨队列留出 |

---

## 6. 附录

### 6.1 评估后未进入主表的候选

| 候选 | 借鉴 | 评估结论 |
|---|---|---|
| 连续 motif + 扩散/能量头 | MAR [70]、VibeVoice（7.5 Hz 连续 tokenizer + next-token diffusion，可生成 90 分钟语音）[42]、CALM [55] | 不建议作为主线：失去离散 motif 与精确似然，与 A4、D1、D2 冲突；Hypnos 代码已有 VAE 模式（`tokenizer.py:186`），新意有限；表征收益不确定。可只在偏生成的实验中用于声学残差 |
| 缩放律研究 | Kaplan / Chinchilla [66]；SpeechLM 的语言能力随算力增长比文本 LLM 慢至多三个数量级 [40]；LSM 的可穿戴数据缩放 [19]；OSF [4]；词表缩放律 [67] | 适合作为主论文的支撑分析：以参数量、夜数、tokenizer 码率（K 与码本大小）、上下文长度为轴，拟合损失与下游表现的关系 |
| 检索增强与个体化 | kNN-LM 不需额外训练即把 Wikitext-103 困惑度降至 15.79（改善 2.9）[62]；VALL-E 用 3 s 声学提示克隆说话人 [43] | 可作为后续工作：用同一被试的往夜记录（SHHS 两次访视、MrOS 多次访视）建立个人基线，计算个体化惊异度（与 D1 结合） |
| FSQ 单码本 | FSQ [72]；X-Codec2 [29] | 已并入 A1，作为语义层的实现选项 |
| μP 超参迁移 | Tensor Programs V [84] | 工具性技术，用于降低缩放实验的调参成本 |

### 6.2 实测脚本输出

`python docs/research/amplitude_probe.py` 的输出（numpy 2.4.6、scipy 1.17.1）：

```text
[1] gain invariance: max |pre(x) - pre(0.3x)| = 0.0

[2] 0.25 Hz respiratory-like signal, event starts at t=600 s
    rows: event duration; cols: true excursion drop -> drop still visible after preprocessing
    dur(s)       30%       50%       90%
        10     25.6%     43.6%     77.8%
        20     26.0%     44.5%     87.1%
        30     22.2%     40.8%     84.0%
        60     18.2%     34.0%     81.3%
       120      9.5%     21.2%     73.6%

[3] slow-wave amplitude decline over 6 h (first vs last hour, p2p envelope)
    raw:            170.5 ->    96.6  (ratio 0.57)
    preprocessed:    3.27 ->    3.76  (ratio 1.15)
```

说明：信号为合成信号，用于说明预处理的数学性质，不代表真实 PSG 上的效应大小。真实数据上的影响需要在阶段 0 用 NSRR 或 DOD-O 数据验证。

### 6.3 参考文献

**睡眠与生理信号**

1. J. F. Carter, L. Tarassenko. Next-Token Prediction Learns Generalisable Representations of Sleep Physiology (Hypnos). arXiv:2606.09605, 2026.
2. R. Thapa et al. A multimodal sleep foundation model for disease prediction. Nature Medicine, 2026.
3. R. Thapa et al. SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals. ICML 2024. arXiv:2405.17766.
4. OSF: On Pre-training and Scaling of Sleep Foundation Models. ICML 2026. arXiv:2603.00190.
5. Stanford Sleep Bench: Evaluating Polysomnography Pre-training Methods for Sleep Foundation Models. arXiv:2512.09591, 2025.
6. SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures. ICML 2026. arXiv:2602.07628.
7. B. Fox et al. A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages (PFTSleep). SLEEP 48(8): zsaf061, 2025. medRxiv 2024.08.02.24311417.
8. Sleep EEG foundation models reveal within-stage microstructure that improves health screening beyond traditional stages. npj Digital Medicine, 2026.
9. J. Pradeepkumar et al. Tokenizing Single-Channel EEG with Time-Frequency Motif Learning (TFM-Tokenizer). ICLR 2026. arXiv:2502.16060.
10. CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model. ICLR 2026. arXiv:2506.09110.
11. W. Jiang et al. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (LaBraM). ICLR 2024. arXiv:2405.18765.
12. W. Jiang et al. NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals. ICLR 2025. arXiv:2409.00101.
13. BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals. NeurIPS 2025. arXiv:2505.18185.
14. REVE: A Foundation Model for EEG, Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects. NeurIPS 2025. arXiv:2510.21585.
15. J. Jin et al. Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model (HeartLang). ICLR 2025. arXiv:2502.10707.
16. K. Zha et al. Physiology as Language: Translating Respiration to Sleep EEG. ICML 2026. arXiv:2602.00526.
17. Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders. arXiv:2605.13930, 2026.
18. eeg-gpt: Next-token generative model of sleep EEG (100 ms tokens). https://github.com/ali77sina/eeg-gpt
19. G. Narayanswamy et al. Scaling Wearable Foundation Models (LSM). ICLR 2025. arXiv:2410.13638.
20. LSM-2: Learning from Incomplete Wearable Sensor Data. arXiv:2506.05321, 2025.
21. Y. Zhang et al. SensorLM: Learning the Language of Wearable Sensors. NeurIPS 2025. arXiv:2506.09108.
22. H. Banville et al. Uncovering the structure of clinical EEG signals with self-supervised learning. Journal of Neural Engineering 18(4): 046020, 2021.
23. American Academy of Sleep Medicine. The AASM Manual for the Scoring of Sleep and Associated Events; R. B. Berry et al. Rules for scoring respiratory events in sleep: update of the 2007 AASM Manual. Journal of Clinical Sleep Medicine 8(5): 597-619, 2012.
24. A. A. Borbély. A two process model of sleep regulation. Human Neurobiology 1: 195-204, 1982; B. A. Mander, J. R. Winer, M. P. Walker. Sleep and Human Aging. Neuron 94(1): 19-36, 2017.
25. R. Vallat, M. P. Walker. An open-source, high-performance tool for automated sleep staging (YASA). eLife 10: e70092, 2021.

**语音与音频**

26. A. Défossez et al. Moshi: a speech-text foundation model for real-time dialogue (Mimi codec). arXiv:2410.00037, 2024.
27. X. Zhang et al. SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models. ICLR 2024. arXiv:2308.16692.
28. Z. Ye et al. Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model (X-Codec). AAAI 2025. arXiv:2408.17175.
29. Z. Ye et al. Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis (X-Codec2). arXiv:2502.04128, 2025.
30. XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs. arXiv:2506.23325, 2025.
31. A. Défossez, J. Copet, G. Synnaeve, Y. Adi. High Fidelity Neural Audio Compression (EnCodec). TMLR 2023. arXiv:2210.13438.
32. N. Zeghidour et al. SoundStream: An End-to-End Neural Audio Codec. IEEE/ACM TASLP, 2022. arXiv:2107.03312.
33. F. Shen et al. Acoustic BPE for Speech Generation with Discrete Tokens. ICASSP 2024. arXiv:2310.14580.
34. A. Baade, P. Peng, D. Harwath. SyllableLM: Learning Coarse Semantic Units for Speech Language Models. ICLR 2025. arXiv:2410.04029.
35. J. Copet et al. Simple and Controllable Music Generation (MusicGen). NeurIPS 2023. arXiv:2306.05284.
36. Kyutai. Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling. arXiv:2509.08753, 2025.
37. M. Han et al. NEST-RQ: Next Token Prediction for Speech Self-Supervised Pre-Training. arXiv:2409.08680, 2024.
38. C.-C. Chiu et al. Self-supervised Learning with Random-projection Quantizer for Speech Recognition (BEST-RQ). ICML 2022. arXiv:2202.01855.
39. W.-N. Hsu et al. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM TASLP, 2021. arXiv:2106.07447.
40. S. Cuervo, R. Marxer. Scaling Properties of Speech Language Models. EMNLP 2024. arXiv:2404.00685.
41. T. A. Nguyen et al. The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. arXiv:2011.11588, 2020.
42. Z. Peng et al. VibeVoice Technical Report. arXiv:2508.19205, 2025.
43. C. Wang et al. Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E). arXiv:2301.02111, 2023.
44. C. Fifty et al. Restructuring Vector Quantization with the Rotation Trick. ICLR 2025. arXiv:2410.06424.
45. D. Lee et al. Autoregressive Image Generation using Residual Quantization (RQ-Transformer). CVPR 2022. arXiv:2203.01941.

**文本与大语言模型**

46. F. Gloeckle et al. Better & Faster Large Language Models via Multi-token Prediction. ICML 2024. arXiv:2404.19737.
47. DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024.
48. Z. Lin et al. Not All Tokens Are What You Need for Pretraining (Rho-1). NeurIPS 2024. arXiv:2404.07965.
49. S. M. Xie et al. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. NeurIPS 2023. arXiv:2305.10429.
50. T. Gao et al. Metadata Conditioning Accelerates Language Model Pre-training (MeCo). ICML 2025. arXiv:2501.01956.
51. A. Pagnoni et al. Byte Latent Transformer: Patches Scale Better Than Tokens. ACL 2025. arXiv:2412.09871.
52. S. Hwang, B. Wang, A. Gu. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net). arXiv:2507.07955, 2025.
53. L. Yu et al. MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers. NeurIPS 2023. arXiv:2305.07185.
54. L. Ren et al. Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling. ICLR 2025. arXiv:2406.07522.
55. C. Shao et al. Continuous Autoregressive Language Models (CALM). arXiv:2510.27688, 2025.
56. H. Huang, Y. LeCun, R. Balestriero. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures. ICLR 2026. arXiv:2509.14252.
57. W. Liang et al. Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models. TMLR 2025. arXiv:2411.04996.
58. Y. Tay et al. UL2: Unifying Language Learning Paradigms. ICLR 2023. arXiv:2205.05131.
59. M. Bavarian et al. Efficient Training of Language Models to Fill in the Middle. arXiv:2207.14255, 2022.
60. E. Hu et al. The Belief State Transformer. ICLR 2025. arXiv:2410.23506.
61. O. Skean et al. Layer by Layer: Uncovering Hidden Representations in Language Models. ICML 2025. arXiv:2502.02013.
62. U. Khandelwal et al. Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020. arXiv:1911.00172.
63. L. Götz et al. Byte Pair Encoding for Efficient Time Series Forecasting. arXiv:2505.14411, 2025.
64. T. Bricken et al. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread, 2023; A. Templeton et al. Scaling Monosemanticity. Transformer Circuits Thread, 2024.
65. L. Gao et al. Scaling and evaluating sparse autoencoders. arXiv:2406.04093, 2024.
66. J. Kaplan et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020; J. Hoffmann et al. Training Compute-Optimal Large Language Models. arXiv:2203.15556, 2022.
67. C. Tao et al. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies. NeurIPS 2024. arXiv:2407.13623.

**视觉、多模态与生物序列**

68. D. Mizrahi et al. 4M: Massively Multimodal Masked Modeling. NeurIPS 2023. arXiv:2312.06647; R. Bachmann et al. 4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities. NeurIPS 2024. arXiv:2406.09406.
69. K. Tian et al. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction (VAR). NeurIPS 2024. arXiv:2404.02905.
70. T. Li et al. Autoregressive Image Generation without Vector Quantization (MAR). NeurIPS 2024. arXiv:2406.11838.
71. A. El-Nouby et al. Scalable Pre-training of Large Autoregressive Image Models (AIM). ICML 2024. arXiv:2401.08541.
72. F. Mentzer et al. Finite Scalar Quantization: VQ-VAE Made Simple. ICLR 2024. arXiv:2309.15505.
73. J. Yu et al. CoCa: Contrastive Captioners are Image-Text Foundation Models. TMLR 2022. arXiv:2205.01917.
74. Qwen Team. Qwen2.5-VL Technical Report. arXiv:2502.13923, 2025.
75. R. Girdhar et al. ImageBind: One Embedding Space To Bind Them All. CVPR 2023. arXiv:2305.05665.
76. E. Simon, J. Zou. InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders. Nature Methods, 2025. arXiv:2412.12101.
77. G. Brixi et al. Genome modelling and design across all domains of life with Evo 2. Nature, 2026. bioRxiv 2025.02.18.638918.

**时间序列与分布外检测**

78. T. Kim et al. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift (RevIN). ICLR 2022.
79. G. Woo et al. Unified Training of Universal Time Series Forecasting Transformers (Moirai). ICML 2024. arXiv:2402.02592.
80. A. F. Ansari et al. Chronos: Learning the Language of Time Series. TMLR 2024. arXiv:2403.07815.
81. X. Shi et al. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. ICLR 2025. arXiv:2409.16040.
82. J. Ren et al. Likelihood Ratios for Out-of-Distribution Detection. NeurIPS 2019. arXiv:1906.02845.
83. J. Serrà et al. Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models. ICLR 2020. arXiv:1909.11480.
84. G. Yang et al. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (μP). arXiv:2203.03466, 2022.
