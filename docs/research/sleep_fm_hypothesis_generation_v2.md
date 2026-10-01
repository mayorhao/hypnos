# 睡眠基础模型的失败模式与创新假设（v2）：以语义 motif 为中心的 12 个方案

> **目标**：在 Hypnos 基础上产出可投顶会（ICML / NeurIPS / ICLR）的创新点。本版聚焦两件事：(1) 系统识别前人 EEG / 睡眠基础模型的失败模式，并逐条给出证据；(2) 从语音、文本、视觉、时间序列、生物序列等领域引入有差异、解决了重要问题、且有潜力成为创新点的方案，形成 12 个可评估的研究假设。
> **用户约束**：复用 per-modality tokenizer 的思想；复用 motif 的概念；token 要有语义，而不是只以重构为主要目标。
> **与 v1 的关系**：v1 见 [`sleep_fm_improvement_proposals.md`](sleep_fm_improvement_proposals.md)（2026-09-28，17 个方案 A1-D2）。v2 不重复 v1 的内容，而是（a）补上 v1 缺少的"领域级失败模式分析"，（b）把用户新增的"语义 token"偏好作为主轴重组方案，（c）纳入 2026 年 5 月至 9 月新出现的工作（SleepFM-2、SleepLM、FAME、NSP、EEG-JEPA、ReLMCodec、LLM-Codec、NCP 等）。v1 中仍然有效的方案在第 5.3 节给出映射。
> **日期**：2026-09-30

**证据来源说明**

- 本会话环境的出口代理拦截 arxiv.org、openreview.net、huggingface.co、nature.com 等域名，只有 github.com 可直接读取。论文层面的数字来自检索引擎对原文的摘要，以及 GitHub README。所有数字在写入论文前必须回原文逐条核对；文中对不确定的数字标注"检索摘要"。
- 标注"实测"的结果由本仓库代码复现，脚本为 [`semantic_vq_probe.py`](semantic_vq_probe.py)（本版新增，只依赖 numpy 与 scipy）和 v1 的 [`amplitude_probe.py`](amplitude_probe.py)。
- "创新性"判断基于 2026 年 9 月 30 日的检索，可能遗漏同期工作；投稿前应对选定方向再做一次定向检索。

**术语**

- **per-modality tokenizer**：每类信号有独立的 SEANet + RVQ tokenizer（`src/hypnos/models/tokenizer/`），把波形变成离散码。
- **motif**：每个模态每 1 s 窗口得到的离散码（Hypnos 中为 K 级 RVQ 码字），作为生理"语言"的基本单元。TFM-Tokenizer 把类似单元称为时频 motif [21]。
- **语义 token**（本文的操作性定义）：一个 token 序列被称为"有语义"，需要同时满足三条可测量的性质：(i) token 与生理状态、事件、临床变量之间的互信息高（例如与 AASM 事件的 purity、NMI）；(ii) token 与主体身份、设备、背景相位等干扰变量之间的互信息低；(iii) token 对后续语言模型是可预测的（下一 token 准确率或注释流的条件 NLL）。只满足"能重构波形"不算有语义。这一定义直接对应第 6 节评测协议（P12）。

---

## 0. 结论摘要

1. **前人失败模式的共同根源**：目前所有主流睡眠 / EEG 基础模型的 token 或表示都由"重构原始信号"或"短窗对比"这两类目标决定，而不是由生理语义决定。2026 年的一批审计工作给出了直接证据：重构目标偏向高功率的非周期性低频成分 [41][35]，表示按主体而非任务聚类 [41][43]，冻结嵌入完全不保留长程时间相关 [46]，掩码或下一步预测可以被局部线索平凡解决 [38][71]，模型规模与下游性能没有一致关系 [47][48]。Hypnos 的 per-modality RVQ tokenizer 只用重构与 commitment 训练（`tokenizer.py:296`），继承了第一类失败模式；本文用合成信号实测，波形 MSE 码本对低功率振荡事件（纺锤波样、alpha、beta）的召回率只有 0.19-0.38，而码与背景相位的 NMI 达 0.35（第 2 节 F1）。
2. **推荐主线**：把"语义 motif"作为论文的中心对象，三层贡献各选一个方案。Tokenizer 层：P1（频谱平衡 + 描述子蒸馏的语义层）与 P2（周期 / 非周期解耦的双流 motif）；语言模型层：P7（motif 之上的下一概念预测）与 P4（主体 / 设备因子化）；评测层：P12（token 语义评测协议与负对照）。P3（可预测性正则）作为 tokenizer 层的强化项，P5 / P6（事件同步与 motif 词）作为第二梯队。暂定叙事：*Semantic motifs: tokenizing sleep physiology for what it means, not what it looks like*。
3. **创新性与可行性综合最高的方案**：P1、P2、P4、P7、P12。全部可以在单机 8 卡、约 5,000 夜的规模上于两个月内完成首轮验证。
4. **不需要重新预训练、可立即出结果的实验**：P12 的诊断套件（用已发布权重计算主体方差占比、频谱审计、DFA 恢复 R²、token purity）；P2 的离线 FOOOF 分解统计；本文的 `semantic_vq_probe.py` 已经跑通。
5. **主要风险**：(a) 最强竞品 SleepFM-2 已用 235,865 夜预训练并在 215 个表型上取得 C-index ≥ 0.75 [4]，本工作在疾病预测上难以正面超越，论文必须以 token 语义、可解释性、生成与零样本能力为差异点，并在同数据、同算力的消融中证明每个组件的贡献；(b) 语义蒸馏与重构冲突（Mimi、XY-Tokenizer 已观察到 [50][53]），需要 split 结构与帕累托曲线；(c) 审稿人可能认为"语义 token"只是多任务监督，P12 的负对照与无标注版本（HuBERT 式自举）是必要的回应。

---

## 1. 评分标准与方法

- **创新性（1-5）**：5 = 检索未见生理领域先例，且迁移非平凡；4 = 仅在单模态或编码器式生理模型中有类似思想，差异清晰；3 = 已知技术首次用于多模态生成式睡眠模型，差异中等；2 = 增量改进；1 = 纯工程。
- **可行性（1-5）**：5 = 改动局部，小规模验证两周内可完成，算力低；4 = 中等改动，小规模重训即可验证；3 = 较大工程量或较多算力；2 = 高风险；1 = 路径不清。
- **共用基础设施不计入单个方案**：训练循环、tokenizer 的判别器与对抗损失需要自行实现（本仓库只含推理代码与模型定义）。NSRR 预处理管线已在 `src/hypnos/data/nsrr.py` 中实现，并且已经计算了本文多个方案需要的描述子：逐秒 EEG / EOG 频带对数功率（`band_log_power`，`nsrr.py:192`）、R 峰与 RR 间期、逐 epoch RMSSD（`rr_descriptors`，`nsrr.py:253`）、归一化尺度 `log_scale`、SpO2 token（`spo2_tokens`，`nsrr.py:309`）、逐秒质控位（`per_second_qc`，`nsrr.py:160`）。
- **与三项偏好的关系**：每个方案标注对 per-modality tokenizer 与 motif 概念是"保持 / 扩展 / 修改"，以及语义性是"直接目标"还是"间接收益"。

---

## 2. 前人研究的失败模式（F1-F12）

每条失败模式给出：现象、证据、涉及的模型、在 Hypnos 代码中的体现、对应方案。

### F1. 重构驱动的 token 编码高功率的非周期性背景与相位，而不是低功率的振荡事件

**证据**

- 对多种重构式 EEG 基础模型的合成信号实验表明，嵌入偏向捕获非周期性（1/f）成分，低估振荡成分，尤其是高频振荡；其后果是表示形成以主体为中心而非以任务为中心的聚类 [41]。
- 后续工作把这一偏置归因于 EEG 的 1/f^α 频谱结构与神经网络的低频偏好之间的相互作用；在掩码自编码器中，ℓ2 重构目标进一步放大失衡：在相对误差相当的情况下，高功率低频成分对损失的贡献不成比例地大。该偏置跨数据规模、模型容量与预训练目标持续存在 [35]。
- 频谱审计框架显示，六种架构在睡眠 / 清醒分类中对非周期成分的依赖极强：拉平 1/f 后平衡准确率下降超过 0.42；对 PTB-XL ECG 也有 0.32-0.36 的下降 [42]。这说明非周期成分携带任务信息（觉醒、年龄、病理），但把它与振荡成分混在一个 token 里会让振荡语义被淹没。
- 语音领域的平行证据：编解码器为波形重构优化，而不是为自回归预测优化，这会把声学噪声注入离散 token 空间并提高语言模型困惑度 [56]；DASB 基准显示语义 token 在判别与生成任务上普遍优于声学 token [67]；视觉领域，重构保真度与多模态可学习性会分离，rFID 不足以评价 tokenizer [88]。

**实测**（`semantic_vq_probe.py`，合成 EEG 样信号，24 个"主体"各 400 s，每个主体有不同的 1/f 指数与幅度；插入低功率振荡事件 alpha、sigma、beta 和一个高幅度 K 复合波样事件；信号经过本仓库 `causal_preprocess_signal` 后切成 1 s 窗；三种 64 码的 k-means 码本，k-means 是均方重构目标下的最优量化器）

| 码本的目标空间 | NMI(码, 事件类别) | NMI(码, 主体) | NMI(码, 背景相位) | 平衡准确率 |
|---|---|---|---|---|
| A. 波形 MSE（重构目标） | 0.097 | 0.078 | **0.347** | 0.405 |
| B. 逐频点标准化的对数谱（频谱平衡） | 0.172 | **0.365** | 0.026 | 0.688 |
| C. 标准化的频带描述子 | **0.207** | 0.281 | 0.030 | **0.775** |

按类别的召回率（多数投票解码，测试主体）：

| 码本 | none | K 复合波样 | alpha | sigma（纺锤波样） | beta |
|---|---|---|---|---|---|
| A. 波形 MSE | 0.244 | 0.963 | 0.246 | 0.381 | 0.190 |
| B. 频谱平衡 | 0.573 | 0.898 | 0.700 | 0.632 | 0.636 |
| C. 频带描述子 | 0.645 | 0.924 | 0.766 | 0.826 | 0.716 |

解读：波形 MSE 码本把码字分配给背景低频成分的相位（NMI 0.35），高幅度事件被完整保留（K 复合波召回 0.96），低功率振荡事件几乎不可分（alpha 0.25、beta 0.19，机会水平 0.20）。这正是 [41][35] 描述的偏置在 1 s 离散 motif 上的表现。频谱平衡的目标（B）解决振荡事件的可分性，但把主体身份编进码字（NMI 0.37，对应 F2）；描述子目标（C）在两者之间取得较好折中。合成实验只说明目标函数的数学性质，真实 PSG 上的效应量需要在阶段 0 用 NSRR 数据复现。

**涉及的模型**：LaBraM、CBraMod、REVE、BENDR、BIOT 等重构式 EEG 模型 [41][35][46]；Stanford Sleep Bench 中的 MAE / DAE 变体 [6]；Hypnos 的 RVQ tokenizer。

**Hypnos 代码中的体现**：`SignalTokenizer.forward`（`tokenizer.py:296`）只返回重构与 commitment 损失；RVQ 第一级量化的是残差方差最大的成分（`quantizer.py`）。

**对应方案**：P1、P2、P3、P12。

### F2. 主体身份陷阱与队列捷径

**证据**

- FMScope 审计三种 EEG 基础模型：冻结表示的主体方差占比是随机高斯零假设的 13-89 倍（12/12 组），微调后在全部 12 组中进一步升高；只有在文献已确认存在跨主体标志物的任务上，微调才提高标签方差（+0.6 至 +8.4 个百分点）[43]。
- 负对照协议：五个预训练编码器在 CAUEEG 正常 / MCI / 痴呆分类上的最优探针 macro-AUROC 为 0.527-0.677，而简单经典特征为 0.734；没有任何任务同时满足四项归因条件，因此不支持"疾病信息跨临床人群迁移"的一般性结论 [44]。
- 隐私审计：一个从某个冻结编码器学到的岭回归属性解码器，经线性桥接后可迁移到其他编码器的留出主体测试集；DP-SGD 与噪声防御无法消除属性通道 [45]。
- sleep2vec（ICLR 2026）指出标准对比学习在医疗数据上按医院而非健康状态分组，提出以人口学、年龄、站点、病史加权负样本的 DASH-aware InfoNCE 来抑制队列捷径 [8]。SleepMaMi 的宏观编码器用人口学引导的对比学习 [7]，也可能把年龄、性别作为捷径。
- 睡眠领域的评论文章指出，训练队列一致偏向老年、单一族裔、合并症人群；疾病预测结论在缺乏人口学消融时难以解释 [14]。

**Hypnos 代码中的体现**：没有任何主体 / 站点不变性约束；channel embedding 与 CRP 分组只处理模态子集鲁棒性（`multimodal_temporal.py:42`）。NTP 目标本身会奖励记住主体特异的背景（F1 的 B 列结果）。

**对应方案**：P2、P4、P8、P12。

### F3. 冻结表示对长程时间相关盲视

**证据**：五个 EEG 基础模型的冻结嵌入都不表示 alpha 包络的 DFA 指数（长程时间相关的标准量）；原始波形模型（REVE、LaBraM、BENDR）既不恢复 DFA 指数也不恢复 1/f 斜率（R² ≤ 0.12），频谱输入模型（CBraMod、BIOT）恢复 1/f（R² 0.59-0.73）但不恢复 DFA；被丢弃的指数在冷启动跨人群迁移中有方向性预测力，而冻结嵌入处于机会水平 [46]。原因是这些模型对短 patch 做重构或对比，再池化为固定向量。

**Hypnos 代码中的体现**：训练窗口为局部滑窗加周期性全局窗（v1 L4），论文最长测试 4096 token（约 68 分钟）[1]；30 s 平均池化（README）丢弃秒级序列的相关结构。

**对应方案**：P5、P7、P9，以及 v1 B2。

### F4. 掩码 / 下一步预测可以被局部线索平凡解决

**证据**

- NSP 指出稳定的位置线索与局部相关性可以让掩码区域在不整合分布式神经上下文的情况下被预测；解决办法是对目标做"身份残差化"（去掉通道身份与相对时间的加性效应）并从上下文中排除目标的空间和时间近邻 [38]。
- Bachmann 与 Nagarajan 证明 teacher forcing 在需要前瞻的任务上会学到 "Clever Hans" 捷径，Transformer 与 Mamba 都失败；多步预测（teacherless）可缓解 [71]。
- 时间序列领域：直接预测未来观测值往往得到弱结构的潜表示，捕获表面噪声而不是连贯的动力学，EIDOS 因此改为潜空间预测 [93]；单步 next-latent 回归识别的是条件均值，不能当作可展开的世界模型（在系数 0.9 的标量自回归上，单步 MSE 0.998，16 步开环误差 5.10）[95]。
- Laya（LeJEPA）与 EEG-JEPA 的受控消融表明，预训练目标而非架构或数据是收益的主要来源：EEG-JEPA 把 14 任务冻结 macro 平衡准确率从 40.49% 提高到 50.42% [36][37]。

**Hypnos 代码中的体现**：逐秒 NTP 对平滑生理信号可以大量依赖局部延续；所有位置、所有 RVQ 级等权交叉熵（`model.py:37`）。

**对应方案**：P3、P7、P8。

### F5. 缺失通道与设备 / 导联异质性

**证据**：OSF 发现现有基础模型对推理时缺失通道泛化失败，通道不变特征学习对预训练至关重要，并用"随机置零 50% 通道 + 连续时间块掩码"的两阶段掩码恢复鲁棒性；但当任务关键通道完全缺失时仍有明显差距 [5]。在 MESA 上，从 EEG 切换到仅 ECG 分期，六个模型的 macro F1 平均下降 0.35 [16]。EEG-Arena 的 20,000 次评测显示通道灵活的模型在多数通道配置下绝对性能更高 [47]。SleepFM-2 把表示扩展到头带、耳内 EEG、腕部 PPG 与体动 [4]。

**Hypnos 代码中的体现**：固定 8 通道（`settings.py:33`），CRP 分组提供模态子集鲁棒性但没有传感器条件化。

**对应方案**：P4（设备因子）、v1 A5。

### F6. 规模不带来一致收益，线性探针不足

**证据**：EEG-Arena（30 个基础模型、25 个监督基线、57 个任务）发现参数量与下游性能没有一致的正相关，但在固定架构下增加预训练数据有持续收益 [47]。EEG-FM-Compass 发现线性探针普遍不足，专用模型仍有竞争力 [48]。睡眠领域的综述指出未微调基础模型的零样本分期表现平平，PSG 嵌入在部分疾病分类上相对人口学基线只有很小的提升 [14][15]。

**解读**：当 token 或表示的语义不足时，规模只放大 F1、F2 的偏置；OSF 与 sleep2vec 报告的缩放收益都建立在通道不变或元数据感知的目标之上 [5][8]。

**对应方案**：P1、P7、P12。

### F7. 用重构指标评价 tokenizer，与语言模型的可学习性脱节

**证据**：ReLMCodec 在 24 种预量化表示上用同一探针量化器与语言模型测试，量化前的音素可分性与下一 token 准确率的 Spearman 相关为 0.911 [55]；LLM-Codec 在 codec 训练中加入 Medusa 式多步未来 token 预测与语义对齐后，token 语言模型的困惑度降低 35 倍，SALMon 语音连贯性提高 12.1 个百分点，同时 Mel 距离改善 5.0% [56]；视觉领域 GigaTok 发现扩大 tokenizer 会提高重构但损害生成，需要语义正则 [83]；VA-VAE 用视觉基础模型对齐损失扩展重构-生成前沿 [82]；MEG 上可学习 tokenizer 与固定离散化在多数指标上相当 [34]，说明重构驱动的可学习 tokenizer 没有学到额外的语义结构。

**Hypnos 代码中的体现**：tokenizer 与 RQ-Transformer 分两阶段训练，tokenizer 的选择只以重建质量为准（README 与 v1 表 1.1）。

**对应方案**：P3、P12。

### F8. 码本坍缩、使用不均与单码本混合时频

**证据**：EEG 基础模型综述指出码本大小、commitment 权重、RVQ 深度与更新策略不加控制会导致码本坍缩或使用不均 [48]；CodeBrain 指出直接照搬图像的单码本 VQ-VAE 会把时域与频域模式混在一起，token 难以与临床可解释的神经事件或节律对齐，因此设计了时频解耦的 TFDual tokenizer [22]；SimVQ 证明坍缩来源于码本的不连续优化，用一个线性层重参数化后 8,192 至 262,144 码都能达到接近 100% 使用率 [89]。

**Hypnos 代码中的体现**：EMA 码本 + 死码替换（`quantizer.py:192`），并有独立的边际熵统计（`marginal_entropy`）用于检测"被强制均匀化"的伪使用率。这是工程上成熟的方案，但不能保证码字有语义。

**对应方案**：P1（语义层用 FSQ / SimVQ）、P2。

### F9. 固定 1 s 网格与生理事件不对齐，token 预算错配

**证据**：ECG 上生理感知的 tokenizer（中位心搏、HeartLang）在四种骨干上的平均 macro-AUC 为 0.893 / 0.889，逐点与 patch 分词为 0.822 / 0.824，序列长度从 1,250 降到 158，峰值显存从 5.21 GB 降到 0.27 GB [31]；心搏同步 token 在 PTB-XL 上 macro AUROC 0.8945，序列长度从 100 降到 11.2 [32]。语音领域的动态帧率 codec 表明"边界是分配决策"，与音节尺度对齐的边界带来更低的池化失真与更好的语义 token 重建 [61]；VARSTok 用自适应聚类与隐式时长编码减少 23% token 并降低 WER [59]。睡眠事件（纺锤波、K 复合波、慢波）典型时长 0.5-3 s，一个 1 s 网格会切开它们 [19]。

**Hypnos 代码中的体现**：所有模态强制共享 `token_duration_sec`（`manifest.py:94-96`）。

**对应方案**：P5、P6。

### F10. 幅度盲区

v1 A2 已实测：滚动 z-score 使 `pre(x)` 与 `pre(0.3x)` 完全相同，6 小时慢波幅度衰减被抹去，30 s 低通气样幅度下降只剩 22%。BandVQ 在 EEG 基础模型中加入了量化的绝对对数功率 token [23]，是这一方向的独立佐证。**对应方案**：P2（非周期流携带绝对功率），v1 A2。

### F11. 封闭标签空间与以分期为中心的评测

**证据**：SleepLM 的动机是现有睡眠模型在封闭标签空间中工作，不能描述、查询或泛化到新的睡眠现象，因此构建了 10 万小时以上的睡眠-文本语料 [9]；Stanford Sleep Bench 显示分期、呼吸暂停、年龄任务上各预训练方法相当，只有死亡与疾病预测上对比学习显著领先（CL-LOO 疾病 C-index 0.743，平均高出重构方法 4.64%），且频域方法一致优于时域 [6]；睡眠领域评论指出评测框架不一致、监督对照稀缺、缺乏人口学消融 [14]。

**对应方案**：P11、P12，以及 v1 D1。

### F12. 生成与似然能力未被评测，语义随算力扩展慢

语音语言模型的语义能力随算力增长比文本慢最多三个数量级 [68]，这是纯 NTP 在离散语音 token 上的已知问题；Hypnos 具备逐 token 似然，但评测只用线性探针（v1 L9）。**对应方案**：P3、P7、P12，以及 v1 D1。

### 失败模式与方案的映射

| 失败模式 | 主要证据 | Hypnos 代码位置 | 方案 |
|---|---|---|---|
| F1 重构偏置 | [41][35][42][56][67][88]，实测 | `tokenizer.py:296` | P1 P2 P3 P12 |
| F2 身份陷阱 | [43][44][45][8][14] | 无不变性约束 | P2 P4 P8 P12 |
| F3 长程相关盲区 | [46] | 训练窗口、30 s 池化 | P5 P7 P9 |
| F4 局部捷径 | [38][71][93][95][36][37] | `model.py:37` | P3 P7 P8 |
| F5 缺失通道 / 设备 | [5][16][47][4] | `settings.py:33` | P4 |
| F6 规模不一致 | [47][48][14][15] | 全局 | P1 P7 P12 |
| F7 重构 ≠ 可学习性 | [55][56][83][82][34] | 两阶段训练 | P3 P12 |
| F8 码本 | [48][22][89] | `quantizer.py:192` | P1 P2 |
| F9 网格错配 | [31][32][61][59][19] | `manifest.py:94` | P5 P6 |
| F10 幅度盲区 | v1 实测，[23] | `pipeline.py:78` | P2 |
| F11 封闭标签 / 评测 | [9][6][14] | 评测协议 | P11 P12 |
| F12 似然未评测 | [68] | `generate.py` | P3 P7 P12 |

---

## 3. 借鉴方案地图

按来源领域列出本版新引入的方案，说明它解决了什么问题、与睡眠的对应关系。v1 已经覆盖的来源（Mimi split RVQ、SpeechTokenizer、X-Codec、BLT、H-Net、MEGABYTE、VAR、MTP、Rho-1、DoReMi、SAE 等）不再重复。

| 领域 | 方案 | 解决的问题 | 与睡眠的对应 | 引用 |
|---|---|---|---|---|
| 语音 | ReLMCodec：量化前的音素可分性预测 NTP 难度（ρ = 0.911），用锚定保持的适配防止重构漂移 | tokenizer 与 LM 目标脱节（F7） | "事件可分性"作为 tokenizer 的设计量 | [55] |
| 语音 | LLM-Codec：codec 训练加入 Medusa 式多步未来 token 头与语义对齐，PPL 降 35 倍 | 同上 | tokenizer 与小型 LM 协同训练 | [56] |
| 语音 | XY-Tokenizer、BiMTokenizer：语义与声学双通道或单塔平衡 | 低码率下语义-声学冲突 | 语义 motif 层与声学残差层 | [53][54] |
| 语音 | Kanade：单流去说话人 token；FACodec：内容 / 韵律 / 细节 / 说话人监督因子化 + 梯度反转 | 说话人身份污染内容 token | 主体身份污染生理 motif（F2） | [57][58] |
| 语音 | VARSTok、DyCAST、FlexiCodec、DTM-Codec：可变帧率与隐式时长编码；边界分析表明有用边界在音节尺度 | 固定帧率与语言单位不对齐 | 纺锤波 / K 复合波 / 呼吸周期尺度的 motif 边界（F9） | [59][60][61][62] |
| 语音 | UniAudio-Token、TaDiCodec：语义 tokenizer 加入通用音频感知 / 文本感知 | 语义 tokenizer 的"声学盲" | 语义 motif 不应丢掉伪迹、鼾声等非事件信息 | [64][65] |
| 语音 | DASB：不经解码直接评测 token | 评价 tokenizer 只看重构 | P12 | [67] |
| 文本 | NCP / ConceptLM：对隐藏状态做 VQ 得到跨多 token 的"概念"词表，NTP + NCP 联合训练；同等性能少 37% 参数或 24% 训练 token；NCP-ArchPreview 扩展到 8.94B 参数、5.73T token | NTP 目标过于局部（F4） | 在秒级 motif 之上学习 epoch 级"睡眠概念" | [73][74] |
| 文本 | HiLP：更高层的抽象潜变量减少潜空间展开的误差累积；LCM：在句子嵌入空间做自回归 | 长视野一致性 | 整夜结构 | [75][76] |
| 文本 | SuperBPE：跨越空白的超词，200k 词表下 token 少 33%，8B 规模 30 个任务一致提升 | 词边界限制 | 跨模态"超词"（如 EEG 觉醒 + EMG 爆发） | [79] |
| 推荐 | TIGER 语义 ID：RQ-VAE 层级 ID，冲突加尾 token | 离散 ID 的语义与唯一性 | motif ID 的层级化与检索 | [81] |
| 视觉 | VA-VAE、GigaTok：用基础模型特征对齐 / 正则 tokenizer 潜空间 | 重构-生成两难 | 用对比式 PSG 编码器（SleepFM-2、OSF）正则 motif 潜空间 | [82][83] |
| 视觉 | SemHiTok：语义码索引纹理子码本 | 理解与生成的特征层级不同 | 语义 motif 索引声学残差子码本 | [84] |
| 视觉 | Semanticist：PCA 式嵌套因果 token，解释方差递减 | token 顺序无结构 | 由粗到细、可截断的 motif 层级 | [86] |
| 视觉 | "tokenizer 即视觉语言"：重构保真度与多模态可学习性分离，I2T 损失是更一致的信号 | tokenizer 评价指标 | 用注释流的条件 NLL 评价 tokenizer | [88] |
| 时序 | EIDOS、LeNEPA：潜空间预测替代观测值预测 | 表面噪声（F4） | P8 | [93][94] |
| 时序 | TS-BPE：motif 合并为变长 token，预测质量平均提升 12.7%（检索摘要） | patch 刚性 | P6 | [91] |
| 时序 | Small Vocabularies, Big Gains：分词配置决定表示容量与稳定性，错配会抵消预训练收益 | 词表设计 | 词表缩放消融 | [92] |
| 生理 | FAME：按频带标准化重构目标并等权，41 个任务中 24 个 SOTA | 低频偏置（F1） | P1 | [35] |
| 生理 | BandVQ：逐频带独立 VQ + 量化绝对对数功率 token + 元数据前缀 | 单码本混合、幅度盲区 | P1 / P2 的最近邻，需要差异化 | [23] |
| 生理 | NeuroRVQ：多尺度特征 + 层级 RVQ + 相位感知损失 | 重构保真 | 声学残差层的参考 | [24] |
| 生理 | PiMT：12 个固定子频带 token | 无需学习的生理先验 | P2 的固定分解基线 | [26] |
| 生理 | NSP、EEG-JEPA、Laya：潜预测 + 身份残差化 + 拓扑分离上下文 | 局部捷径、身份 | P8 | [38][37][36] |
| 生理 | sleep2vec DASH-InfoNCE | 队列捷径 | P4 | [8] |
| 生理 | SleepLM：多层级睡眠字幕 + 对比 / 字幕 / 重构联合目标，ICML 2026 spotlight | 封闭标签空间 | P11 | [9] |
| 生理 | SSSM：276,404 个专家标注的睡眠事件，7 类，采样点级语义分割 | 事件词表缺失 | P5 的边界监督与 P12 的评测 | [19] |
| 生理 | 频谱审计、FMScope、负对照协议、DFA 恢复 | 评测缺乏负对照 | P12 | [42][43][44][46] |
| 生理 | Agentic 发现：网络级生理耦合下降与帕金森（HR 1.48）、阿尔茨海默（HR 1.38）相关 | 跨模态耦合是临床量 | P10 | [18] |

---

## 4. 方案详述（P1-P12）

每个方案的结构：假设、借鉴来源、针对的失败模式、Hypnos 现状、方案、与三项偏好的关系、创新性、可行性、可行性证据、最小验证实验、风险、与 v1 的关系、参考文献。

### P1. 频谱平衡 + 描述子蒸馏的语义层 motif tokenizer

**假设 H1**：如果 per-modality tokenizer 的第一级（语义层）以"按频带标准化的时频目标 + 逐秒生理描述子"为训练目标，而声学残差层保留波形重构，则语义层 motif 对 AASM 事件的 purity 与 NMI 显著高于重构驱动的 RVQ 第一级，且在同码率下下游任务不降。

**借鉴来源**：FAME 的按频带标准化、等权重构目标 [35]；Mimi / SpeechTokenizer 的语义蒸馏与 split 结构 [50][51]（v1 A1）；BandVQ 的逐频带 VQ [23]；CodeBrain 的时频解耦 [22]；X-Codec2 的 FSQ 大码本 [52]；SimVQ [89]。

**针对**：F1、F6、F8。

**Hypnos 现状**：`tokenizer.py:296` 的 `forward` 只有重构与 commitment；`nsrr.py:192` 已经算好逐秒频带对数功率，`nsrr.py:253` 已有 RR 与 RMSSD，尚未被任何损失使用。

**方案**

- 语义层：每模态每秒一个 FSQ 或 SimVQ 码（码本 4k-64k）。训练目标三项：(a) FAME 式频谱平衡重构：预测各频带的标准化时频活动，各频带损失等权，而不是波形 ℓ2；(b) 描述子回归 / 蒸馏：EEG / EOG 用五个频带对数功率与纺锤波 / 慢波检测概率（YASA），EMG 用 RMS，ECG 用 RR 与形态类别，呼吸用呼吸率与努力幅度；(c) 可选的教师蒸馏：对比式 PSG 编码器（SleepFM-2、OSF 已公开权重）的逐 epoch 特征，作为 VA-VAE 式的潜空间对齐项 [82]。
- 声学层：并行的 K-1 级 RVQ 保留波形重构与判别器损失（Mimi split）。
- 语义-声学接口用 SemHiTok 的方式：语义码索引声学子码本，使 depth transformer 先生成语义码，再生成条件残差 [84]。
- 消融维度：无蒸馏 / 只 (a) / 只 (b) / (a)+(b) / (a)+(b)+(c)；语义码本大小；HuBERT 式自举（用模型自身的 temporal context 聚类作为伪标签，避免依赖手工描述子）[66]。

**与三项偏好**：per-modality tokenizer 保持；motif 保持 1 s；语义是直接目标。

**创新性 4/5**：FAME 与 BandVQ 是编码器式单模态 EEG 工作；本方案把频谱平衡目标放进多模态生成式模型的离散 motif 层，并系统比较"频谱平衡"、"描述子"、"对比教师"、"自举"四种语义来源，检索未见。与 BandVQ 的差异：BandVQ 每频带独立 VQ，token 数随频带数线性增长；本方案每秒一个语义码，语义来自目标而非拆分。

**可行性 5/5**：只改 tokenizer 损失，描述子已在 `nsrr.py` 中离线计算。

**可行性证据**：FAME 在 OmniEEG-Bench 41 个任务中 24 个 SOTA [35]；本文实测 C 列（描述子目标）把纺锤波样事件召回从 0.38 提高到 0.83；Mimi 蒸馏显著改善 ABX 而 split 结构恢复重构质量（v1 引用）；X-Codec2 单层 FSQ 码本利用率约 99%（v1 引用）。

**最小验证实验**：约 2,000 夜子集重训 EEG 与 ECG tokenizer；指标：语义层码与分期 / 事件的 NMI 与 purity，DASB 式无解码线性探针，重构 SNR 与频带误差，"重构-语义"帕累托曲线。再训练 50M-100M 的 RQ-Transformer 比较分期 κ、觉醒 F1、AHI 误差。

**风险**：描述子教师把已知生理偏见固化进词表；对策是自举分支与频谱平衡分支不依赖手工特征。

**与 v1 的关系**：升级 A1（新增频谱平衡目标、SemHiTok 接口、实测证据）。

**参考文献**：[22][23][35][50][51][52][66][82][84][89]

### P2. 周期 / 非周期解耦的双流 motif

**假设 H2**：把每秒信号的非周期成分（1/f 斜率、偏移、绝对功率）与周期成分（振荡 motif）分成两条 token 流，能同时降低 motif 与主体身份的互信息、提高与事件的互信息，并恢复 F10 的幅度信息；非周期流本身对年龄、觉醒、病理有独立预测力。

**借鉴来源**：非周期 / 周期分解（specparam / FOOOF 类方法）与频谱审计发现的任务依赖性非周期依赖 [42]；非周期偏置的成因分析 [41][35]；FACodec 把韵律、内容、细节、说话人分成不同 RVQ 流 [58]；BandVQ 的绝对对数功率 token [23]；PiMT 的固定子频带 token [26]；EnCodec 的 scale 侧信道（v1 A2）。

**针对**：F1、F2、F10。

**Hypnos 现状**：滚动 z-score 去掉绝对幅度，`pipeline.py:78` 丢弃 `mu_track`、`std_track`；tokenizer 输入混合非周期与周期成分。

**方案**

- 非周期流（低速率，每 2-5 s 一个 token）：逐窗拟合 1/f 斜率与偏移，加上 `nsrr.py` 已有的 `log_scale`，量化为 64-256 档或 FSQ；EEG、EOG、EMG、ECG 各一条。
- 周期流（1 s motif）：tokenizer 输入为去除拟合非周期成分后的残差（或在时频域减去拟合的 1/f 曲线），再走 P1 的语义层与声学层。
- 语言模型：两条流作为同一模态的两个 depth 级别（先非周期后周期），或作为独立模态参与轴向注意。
- 分析：非周期流的线性探针（年龄、睡眠 / 清醒、AHI）；周期流对主体身份的 NMI 相对单流的下降。

**与三项偏好**：per-modality tokenizer 扩展（每模态两条流）；motif 修改为"周期 motif + 非周期状态"；语义是直接目标。

**创新性 4/5**：频谱审计与非周期偏置论文只做诊断 [41][42]，FAME 做频带平衡 [35]，BandVQ 加绝对功率 [23]；把"非周期 / 周期"作为生成式模型的显式 token 分解，检索未见。生理动机明确：非周期斜率随年龄、觉醒、麻醉与病理变化，是独立的临床量。

**可行性 5/5**：分解可以离线完成（specparam 逐窗拟合，或用线性回归拟合对数谱）；tokenizer 与 RQ-Transformer 的改动局部。

**可行性证据**：拉平 1/f 使睡眠 / 清醒分类平衡准确率下降 0.42 以上 [42]，说明非周期成分携带的信息足以支撑一条独立流；本文实测 B 列显示频谱目标会把主体身份编进码字（NMI 0.37），分离非周期成分是最直接的对策；FACodec 的因子化在语音中被广泛复现 [58]。

**最小验证实验**：用已发布 Hypnos 权重与 NSRR 子集，先做离线统计：非周期斜率与年龄、分期的相关；再训练"单流 vs 双流"小模型，比较周期 motif 的主体 NMI、事件 purity，以及 N3 召回、低通气检测、年龄回归。

**风险**：逐秒拟合 1/f 在 1 s 窗上噪声大，需要 2-5 s 窗或因果平滑；非周期流可能成为设备捷径（放大器、阻抗），需与 P4 的设备因子联合。

**与 v1 的关系**：吸收 A2（尺度 token）并给出更强的生理动机；与 A1 互补。

**参考文献**：[23][26][35][41][42][58]

### P3. 可预测性正则的 tokenizer（tokenizer 与语言模型协同训练）

**假设 H3**：在 tokenizer 训练中加入"小型因果语言模型对下一秒语义码的预测损失"（Medusa 式多步头）与"量化前特征的事件可分性约束"，得到的 motif 对大模型更可预测，同码率下 RQ-Transformer 的 NLL 更低、表示更好，而重构质量不明显下降。

**借鉴来源**：LLM-Codec [56]；ReLMCodec 的"预量化结构决定可预测性"与 PAPA 适配 [55]；"tokenizer 即视觉语言"对可学习性指标的强调 [88]；GigaTok 的语义正则 [83]。

**针对**：F4、F7、F12。

**Hypnos 现状**：两阶段训练，tokenizer 训练时看不到语言模型。

**方案**

- 在 tokenizer 训练图中挂一个 10M-20M 参数的因果 Transformer，在语义层码上做 NTP 与多步（+1、+5、+30 s）预测，梯度经 STE 或 rotation trick（`quantizer.py` 已支持）回传到编码器；权重随训练退火，避免 tokenizer 把信息"藏"起来以降低 LM 损失（用重构损失与描述子损失作为锚，对应 ReLMCodec 的 anchor-preserving）。
- 预量化可分性度量：用 NSRR 事件标注计算量化前特征的事件类别可分性（Fisher 比或 kNN purity），作为 tokenizer 选型指标，检验它是否像语音中一样与 NTP 准确率强相关。
- 模态间：多步头的目标可以包含其他模态的语义码，把跨模态可预测性也作为 tokenizer 的设计量。

**与三项偏好**：per-modality tokenizer 保持；motif 保持；语义是直接目标（可预测性是语义的第 (iii) 条性质）。

**创新性 4/5**：语音领域 2026 年才出现 [55][56]；生理信号中未见。差异在于生理信号的"语义"没有音素那样的离散真值，需要用事件标注与自举聚类替代。

**可行性 4/5**：训练图变复杂，需要平衡三类损失；小型 LM 的开销可控。

**可行性证据**：LLM-Codec 的 PPL 降低 35 倍且重构不降 [56]；ReLMCodec 的 ρ = 0.911 [55]；Hypnos 代码已有 rotation trick 与 STE 两种梯度路径。

**最小验证实验**：固定 RQ-Transformer 架构与数据，比较"无 LM 正则 / 单步 / 多步"三种 tokenizer 训练出的模型的 NLL、线性探针与零样本似然检测（v1 D1）。

**风险**：tokenizer 可能学到过于平滑、丢失事件细节的码；需要监控事件召回与重构 SNR。

**与 v1 的关系**：新方案；与 C1（token 选择）互补。

**参考文献**：[55][56][83][88]

### P4. 主体 / 设备因子化的去身份 motif

**假设 H4**：把主体与设备信息显式地分配给每夜一次的"身份 token"（或连续嵌入），并用梯度反转、DASH 式负样本加权或身份残差化把它从逐秒 motif 中移除，则 motif 的主体方差占比（FMScope 指标）显著下降，跨队列迁移与负对照通过率提高，而分期与事件检测不降。

**借鉴来源**：FACodec 的说话人嵌入 + 梯度反转 [58]；Kanade 的单流去说话人 token [57]；sleep2vec 的 DASH-aware InfoNCE [8]；NSP 的身份残差化 [38]；FMScope 的五项诊断 [43]；负对照协议 [44]；MeCo 元数据前缀（v1 C2）。

**针对**：F2、F5。

**Hypnos 现状**：无任何主体 / 站点不变性约束；channel embedding 可选（`model.py:332`）。

**方案**

- 每夜一个"身份槽"：由前 5 分钟信号经小编码器得到，或由队列、设备、年龄段元数据前缀构成（训练前段使用，冷却阶段去掉）。
- tokenizer 语义层加梯度反转的主体分类头（训练集主体 ID）与设备分类头；语言模型侧对 temporal context 加 NSP 式身份残差化：从潜预测目标中减去按主体、通道估计的加性分量。
- 评测：FMScope 的主体方差占比、主体轴擦除后性能变化、留一队列迁移、负对照（标签置换、随机初始化、经典特征基线）。

**与三项偏好**：per-modality tokenizer 保持；motif 保持；语义是直接目标（第 (ii) 条性质）。

**创新性 4/5**：sleep2vec 在对比框架中处理队列捷径 [8]，NSP 在掩码框架中做身份残差化 [38]；在离散 motif 与生成式模型中做主体 / 设备因子化，并用 FMScope 作为训练目标的评价指标，检索未见。

**可行性 4/5**：梯度反转与元数据前缀实现简单；身份残差化需要按主体统计目标分量，工程量中等。

**可行性证据**：FACodec 通过替换说话人嵌入即可换声，证明因子化在低码率离散流上可行 [58]；Kanade 只用 600 小时与 120M 参数达到说话人解耦 SOTA [57]；FMScope 显示主体方差占比是可测且可比较的量 [43]。

**最小验证实验**：在 2,000 夜、3 个队列上训练"无约束 / 梯度反转 / 残差化"三种模型，报告 FMScope 指标、留一队列分期 κ、年龄回归（预期身份去除后年龄信息应转移到身份槽）。

**风险**：过度去身份会抹掉与疾病相关的个体特征；身份槽必须保留这部分信息，并在疾病预测中允许使用。

**与 v1 的关系**：扩展 C2（MeCo）；新增不变性目标与诊断。

**参考文献**：[8][38][43][44][57][58]

### P5. 事件同步的可变速率 motif

**假设 H5**：让 motif 边界由信号内容决定（事件起止、呼吸周期、心搏），并用隐式时长编码把时长并入 token，则同等 token 预算下事件检测 F1 与长上下文任务更好，token 数减少 20-40%。

**借鉴来源**：VARSTok 的密度峰聚类分段与隐式时长编码 [59]；DyCAST 的软对齐可调边界 [60]；动态帧率边界分析 [61]；FlexiCodec、DTM-Codec [62]；H-Net 动态分块 [78]（v1 A4 路线 2）；SSSM 的 7 类、276,404 个事件标注 [19]；ECG 心搏同步 token 的效率证据 [31][32]。

**针对**：F9、F3。

**Hypnos 现状**：固定 1 s 网格（`manifest.py:94-96`）。

**方案**

- 边界预测器：在 P1 语义层特征上学习边界（BLT 式熵阈值、VARSTok 式相似度聚类，或用 SSSM 事件起止做弱监督）；每个 motif 附带时长码。
- EEG / EOG / EMG 采用事件同步边界，ECG 用心搏，呼吸用呼吸周期，回退到固定窗以保证鲁棒性。
- 位置编码改为绝对时间（v1 A3），跨模态注意按时间窗。
- 分析：边界与专家事件的对齐 F1（[61] 的分析范式），token 预算分布随分期的变化。

**与三项偏好**：per-modality tokenizer 保持；motif 修改为"内容决定边界"，这是对前提的有意修改，论文需保留 1 s 版本作为对照；语义间接受益。

**创新性 4/5**：ECG 单模态与语音已有 [31][32][59]；多模态睡眠中的事件同步离散 motif 未见。

**可行性 3/5**：变长序列的批处理与缓存、CRP 分组、modality attention 需要重写。

**可行性证据**：VARSTok 少 23% token 且 WER 更低 [59]；ECG 生理感知分词序列长度降至 1/8、AUC 提高 7 个百分点 [31]；SSSM 事件资源可用于边界监督 [19]。

**最小验证实验**：只在 EEG 上启用可变速率，其他模态不变；比较觉醒与纺锤波检测 F1、token 数与分期 κ。

**风险**：边界误差传播；与 P6 功能重叠，建议二选一或分阶段。

**与 v1 的关系**：合并 A3 与 A4 路线 2，并给出语音领域的新证据。

**参考文献**：[19][31][32][59][60][61][62][78]

### P6. motif 词与跨模态超词

**假设 H6**：在语义层 motif 序列上学习 BPE 词表，并允许合并跨模态同时出现的 motif 对（SuperBPE 式跨边界合并），得到的"生理词"具有 Zipf 分布、与 AASM 事件对应，并在同算力下延长有效上下文，改善 OSA 与疾病预测。

**借鉴来源**：SuperBPE [79]；TS-BPE [91]；Acoustic BPE [70]；TIGER 语义 ID 的层级化与冲突处理 [81]；Small Vocabularies, Big Gains 的词表消融方法 [92]。

**针对**：F9、F3。

**方案**：v1 A4 路线 1 的基础上新增两点：(a) 跨模态合并：把同一秒不同模态的语义码视为相邻符号，允许 BPE 合并出"EEG 觉醒 + EMG 爆发"、"呼吸暂停结束 + 心率上升"这类超词；(b) 词 ID 采用 TIGER 式层级语义 ID（粗类、细类、消歧尾码），便于检索与可解释分析。

**与三项偏好**：per-modality tokenizer 保持；motif 扩展为词；语义间接受益。

**创新性 3/5**；**可行性 4/5**。

**可行性证据**：SuperBPE 在 8B 规模 30 个任务一致提升、推理效率提高 27% [79]；TS-BPE 在多个时序基础模型上有一致收益（检索摘要 [91]）。

**最小验证实验**：报告压缩倍数、词与事件的对应、跨模态超词的频率按分期与疾病的差异。

**风险**：词分布跨队列漂移；事件边界精度需局部模型补偿。

**与 v1 的关系**：升级 A4 路线 1。

**参考文献**：[70][79][81][91][92]

### P7. motif 之上的下一概念预测（NCP）

**假设 H7**：在 RQ-Transformer 的 temporal context 上做向量量化得到跨多秒的"睡眠概念"词表，并联合训练 NTP 与下一概念预测，可以缓解逐秒 NTP 的局部延续捷径，改善需要慢动态的任务（分期转换、OSA、疾病），并得到一个可解释的离散概念词典。

**借鉴来源**：ConceptLM / NCP [73]；NCP-ArchPreview [74]；HiLP [75]；LCM [76]；Pitfalls of NTP 的 teacherless 训练 [71]；v1 B2（层级生成）与 B3（MTP）。

**针对**：F3、F4、F6。

**Hypnos 现状**：只有逐秒 NTP；`embeddings["1s"]`（`model.py:630`）是唯一的表示读出。

**方案**

- 概念模块：对每 30 s（或每个 P5 / P6 的词）窗口内 temporal context 的池化向量做乘积量化（NCP-ArchPreview 的做法），概念词表 4k-16k；概念预测头预测下一 epoch 的概念码；预测的概念反馈到 token 级作为条件。
- 与 v1 B2 的关系：B2 是显式的两级生成器；P7 是在单一模型中加辅助离散目标，实现更轻，且概念词典可直接分析（与分期、事件、疾病的对应）。
- teacherless 变体：在 lookahead 任务（预测 60 s 后的事件）上加 dummy token 训练。

**与三项偏好**：per-modality tokenizer 保持；motif 扩展为"概念"层级；语义是直接目标（概念词典）。

**创新性 4/5**：NCP 在 2026 年才出现于文本 [73][74]；生理信号中未见。生理动机：睡眠有天然的多尺度层级（事件、epoch、周期、整夜）。

**可行性 4/5**：只增加一个 VQ 模块与预测头；NCP-ArchPreview 报告更新 17M 参数的 VQ 模块即可做领域适配 [74]。

**可行性证据**：ConceptLM 同性能少 37% 参数或 24% token [73]；NCP-ArchPreview 学习速度接近两倍并在推理与数学基准上更高（检索摘要）[74]；Hypnos 的 OSA 结果随上下文增长持续改善 [1]。

**最小验证实验**：同算力比较 NTP / NTP+MTP / NTP+NCP，指标为分期转换处的 κ、OSA 分级、疾病 C-index；概念词典与分期的 purity。

**风险**：概念 VQ 可能坍缩到分期这种粗粒度；用码本大小与熵正则控制。

**与 v1 的关系**：与 B2、B3 互补，可作为 B2 的轻量替代。

**参考文献**：[1][71][73][74][75][76]

### P8. 身份残差化的潜变量预测辅助目标

**假设 H8**：在 NTP 之外加入对未来 epoch 潜表示的预测（EMA 目标编码器），且目标经身份残差化、上下文排除近邻，可以在不改变生成能力的前提下提高疾病与年龄任务上的表示质量。

**借鉴来源**：NSP [38]；EEG-JEPA 的目标内容 / 支持 / 深度三维设计 [37]；Laya [36]；EIDOS [93]；LeNEPA 的 SIGReg 正则 [94]；多模态 EEG 世界模型 [39]；v1 B3 的 JEPA 项。

**针对**：F4、F2。

**方案**：潜预测头预测 t+30 s、t+300 s 的 EMA 目标；目标减去按主体与通道估计的加性分量；预测器的可见上下文排除目标前 5 s；用 SIGReg 而非教师网络稳定潜空间以减少超参。注意 [95] 的警告：单步潜预测不是可展开的世界模型，因此本方案只作为表示辅助目标，不用于长程采样。

**与三项偏好**：三者保持；语义间接受益。

**创新性 3/5**；**可行性 5/5**。

**可行性证据**：EEG-JEPA 14 任务冻结平衡准确率 +10 个百分点 [37]；Laya 用 10% 数据超过 LaBraM 与 REVE 的冻结探针 [36]；LeNEPA 在 2-5k 步内达到 80% 收益 [94]。

**最小验证实验**：NTP vs NTP + 潜预测（有 / 无残差化），报告 FMScope 主体方差占比与疾病 C-index。

**与 v1 的关系**：升级 B3 的 JEPA 项。

**参考文献**：[36][37][38][39][93][94][95]

### P9. 长程相关的慢流 token

**假设 H9**：把每 epoch 的长程相关量（alpha / sigma 包络的 DFA 指数、包络自相关、非周期斜率的漂移）作为一条低速率离散流加入模型，模型的冻结表示能恢复 DFA 指数（R² 从接近 0 提高到 0.5 以上），并改善跨人群冷启动迁移。

**借鉴来源**：LRTC 盲视的诊断 [46]；SpO2 每秒 token 的多速率先例（`nsrr.py:309`）；Qwen2.5-VL 绝对时间位置（v1 A3）。

**针对**：F3。

**方案**：慢流 token 每 30 s 一个，由过去 5-10 分钟的因果窗计算；作为额外模态进入轴向注意（`modality_mask` 已支持缺失流）。评测按 [46] 的协议：从冻结嵌入线性回归 DFA 指数与 1/f 斜率。

**与三项偏好**：per-modality tokenizer 扩展（新模态族）；motif 扩展；语义是直接目标（慢流的每个 token 有明确物理意义）。

**创新性 3/5**：技术简单，价值在于直接回应 [46] 的诊断并给出跨人群迁移的证据。**可行性 5/5**。

**最小验证实验**：已发布权重上先复现 [46] 的 DFA 恢复 R²（预期接近 0），再训练加慢流的小模型对比。

**风险**：慢流可能成为年龄捷径；与 P4 联合评估。

**参考文献**：[46]

### P10. 跨模态耦合 motif 词表

**假设 H10**：把同一时刻多模态之间的耦合状态（心肺耦合相位、EEG-EMG 觉醒同步、呼吸事件与血氧下降的滞后）量化为离散"耦合 motif"，并作为 depth 生成的一级或独立流，可以捕获 [18] 报告的与神经退行性疾病相关的网络级耦合下降，改善疾病预测与可解释性。

**借鉴来源**：Agentic 发现研究中网络级耦合与帕金森（HR 1.48）、阿尔茨海默（HR 1.38）的关联 [18]；心肺耦合的睡眠医学文献；SleepFM 的留一模态对比 [2]；v1 B1 的跨模态联合 depth。

**针对**：F4（跨模态信息只经由上一秒间接建模）与临床价值。

**方案**：在 modality attention 层的输出上加一个跨模态耦合头，对模态对（EEG-ECG、ECG-呼吸、EEG-EMG）的联合表示做 VQ 得到耦合码；耦合码作为每秒"和弦"的第一级（v1 B1）；无标注下用跨模态对比（LOO-CL）作为耦合头的训练信号，有标注时用事件对齐监督。

**与三项偏好**：per-modality tokenizer 保持，新增跨模态 tokenizer；motif 扩展为耦合 motif；语义是直接目标。

**创新性 4/5**：耦合分析在睡眠医学中是经典量，但作为生成式模型的离散词表未见。

**可行性 3/5**：耦合头的设计与评测需要生理学验证；建议先用离线心肺耦合指标做上限估计。

**可行性证据**：[18] 的 HR 数值；Stanford Sleep Bench 中跨模态对比在疾病预测上领先 [6]，暗示跨模态关系携带疾病信息。

**最小验证实验**：离线计算心肺耦合与 EEG-EMG 同步指标，检验其对 MrOS / SHHS 中神经退行与心血管结局的预测力；再实现耦合头并比较疾病 C-index。

**风险**：耦合码可能被伪迹（ECG 伪迹进入 EEG）主导；需要伪迹检测与质控位（`nsrr.py:160`）。

**参考文献**：[2][6][18]

### P11. 语言接地的 motif 码本

**假设 H11**：把 motif 码本嵌入与自然语言描述（由 NSRR 标注模板生成，或复用 SleepLM 的多层级字幕管线）对齐，可以让离散 motif 支持零样本事件查询与开放词表的事件定位，同时不损害生成能力。

**借鉴来源**：SleepLM [9]；TaDiCodec 的文本感知 tokenizer [65]；UniAudio-Token [64]；NeuroLM 的文本对齐 tokenizer [27]；TokLIP 把 VQ token 语义化 [85]。

**针对**：F11。

**方案**：在 P1 语义层上加一个对比头，把每秒 motif 嵌入与该秒所在事件 / 分期的模板文本嵌入对齐；语言模型侧加注释流（v1 B6）。区别于 SleepLM：SleepLM 是连续编码器加字幕生成，本方案让离散 motif 本身携带文本可查询的语义，并保留自回归生成与似然。

**与三项偏好**：三者保持；语义是直接目标。

**创新性 3/5**：SleepLM 与 NeuroLM 已做文本对齐；离散生成式 motif 的文本接地是增量。**可行性 4/5**：NSRR XML 提供事件与分期，模板字幕可自动生成。

**可行性证据**：SleepLM 用 10 万小时以上的睡眠-文本数据实现零样本与少样本任务（ICML 2026 spotlight）[9]；TaDiCodec 在 6.25 Hz 单层码本上保持 WER 与说话人相似度 [65]。

**最小验证实验**：零样本事件检索（文本查询 → 秒级定位）的 AUPRC；与 SleepLM 公开模型比较。

**风险**：模板字幕语义贫乏；可与 SleepLM 管线合作或复用其开源代码。

**参考文献**：[9][27][64][65][85]

### P12. token 语义评测协议与负对照

**假设 H12**：一套不依赖解码器、包含负对照的 token 语义评测协议，能够比重构指标更好地预测下游表现，并让"语义 motif"的主张可证伪。

**借鉴来源**：DASB 的无解码评测 [67]；FMScope [43]；负对照协议 [44]；频谱审计 [42]；DFA 恢复 [46]；ZeroSpeech 的 sWUGGY / sBLIMP（v1 D1）[69]；"tokenizer 即视觉语言"用 I2T 损失评价 tokenizer [88]；ReLMCodec 的预量化可分性 [55]；Cuervo 与 Marxer 的损失-能力相关性 [68]。

**针对**：F2、F6、F7、F11、F12。

**方案**（指标分四组）

1. 语义性：motif 与 AASM 事件 / 分期的 purity、NMI；SSSM 7 类事件 [19] 上的秒级检索 AUPRC；PhysioBLIMP 最小对（v1 D1）。
2. 不变性：FMScope 主体方差占比与主体轴擦除；设备 / 站点可预测性；频谱审计的 1/f 拉平敏感度。
3. 可学习性：语义码的下一 token 准确率；注释流的条件 NLL（对应 I2T 损失）；预量化事件可分性与 NTP 准确率的相关。
4. 负对照：标签置换、随机初始化编码器、经典特征基线（YASA 特征 + GBM）、人口学基线；跨队列留一。

同时报告重构 SNR 与频带误差，画"语义-重构"帕累托图。协议以脚本形式发布，先在 Hypnos 已发布权重、OSF、SleepFM-2、TFM-Tokenizer、CodeBrain 上跑一遍作为基线。

**与三项偏好**：三者保持；语义是直接目标。

**创新性 3/5**（评测类工作在 EEG 领域已有多篇 [43][44][47]，但针对离散 motif 的语义评测协议未见）；**可行性 5/5**（可用已发布权重立即开始）。

**最小验证实验**：对 Hypnos 权重与两个对比式模型运行全部四组指标；报告哪些指标与分期 / 疾病任务的下游表现相关。

**风险**：指标过多导致论文分散；建议正文只保留与下游相关性最强的 4-6 个。

**与 v1 的关系**：扩展 D1；吸收 B7(a) 的逐层读出分析。

**参考文献**：[19][42][43][44][46][47][55][67][68][69][88]

---

## 5. 评分矩阵、推荐组合与实验计划

### 5.1 评分矩阵

| # | 方案 | 针对失败模式 | per-modality tokenizer | motif | 语义 | 创新 | 可行 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| P1 | 频谱平衡 + 描述子蒸馏的语义层 | F1 F6 F8 | 保持 | 保持 | 直接 | 4 | 5 | 高 |
| P2 | 周期 / 非周期双流 motif | F1 F2 F10 | 扩展 | 修改 | 直接 | 4 | 5 | 高 |
| P3 | 可预测性正则的 tokenizer | F4 F7 F12 | 保持 | 保持 | 直接 | 4 | 4 | 高 |
| P4 | 主体 / 设备因子化 | F2 F5 | 保持 | 保持 | 直接 | 4 | 4 | 高 |
| P5 | 事件同步可变速率 motif | F9 F3 | 保持 | 修改 | 间接 | 4 | 3 | 中 |
| P6 | motif 词与跨模态超词 | F9 F3 | 保持 | 扩展 | 间接 | 3 | 4 | 中 |
| P7 | 下一概念预测 | F3 F4 F6 | 保持 | 扩展 | 直接 | 4 | 4 | 高 |
| P8 | 身份残差化潜预测 | F4 F2 | 保持 | 保持 | 间接 | 3 | 5 | 中（消融） |
| P9 | 长程相关慢流 token | F3 | 扩展 | 扩展 | 直接 | 3 | 5 | 中 |
| P10 | 跨模态耦合 motif | 临床 | 扩展 | 扩展 | 直接 | 4 | 3 | 中 |
| P11 | 语言接地的 motif | F11 | 保持 | 保持 | 直接 | 3 | 4 | 中 |
| P12 | 语义评测协议与负对照 | F2 F6 F7 F11 F12 | 保持 | 保持 | 直接 | 3 | 5 | 高 |

### 5.2 推荐组合：Semantic motifs

三项核心贡献：

1. **Tokenizer**：P1 + P2（P3 作为强化项）。叙事：Hypnos 的 per-second motif 由重构决定，编码背景相位而不是振荡事件（实测），并把非周期背景与主体身份混入码字；提出"非周期状态 + 语义 motif + 声学残差"三通道的 per-modality tokenizer，语义层由频谱平衡目标、生理描述子和可预测性正则共同决定。
2. **模型**：P7（下一概念预测）+ P4（主体 / 设备因子化）。叙事：在秒级 motif 之上学习离散的睡眠概念词典，并把身份信息移到每夜一次的身份槽。
3. **评测**：P12 的四组指标 + v1 D1 的零样本似然任务；疾病预测按 Stanford Sleep Bench 与 SleepFM-2 的协议报告，并做人口学基线消融。

第二梯队：P5 或 P6（二选一）延长上下文；P9 作为对 [46] 的直接回应；P10、P11 作为扩展章节或后续工作。v1 中的 B2、B6、D2 仍可作为补充贡献。

结构示意：

```text
EDF / 多导原始信号
  |
  v
per-modality tokenizer (P1 + P2, 可选 P3)
  每模态每 2-5 s: [非周期状态 a_t: 1/f 斜率, 偏移, 绝对功率]
  每模态每 1 s:   [语义 motif m_t: 频谱平衡 + 描述子 + 可预测性] -> 索引 [声学残差 r_t(1..K-1)]
  |
  v  (可选 P5 / P6: 事件同步边界或 BPE 词)
RQ-Transformer: 轴向时间 / 模态注意 + depth (a_t -> m_t -> r_t)
  身份槽 (P4): 每夜一次的主体 / 设备嵌入, motif 经梯度反转去身份
  概念模块 (P7): temporal context 的乘积量化 -> 下一概念预测
  [可选] 潜预测辅助 (P8), 慢流 (P9), 耦合 motif (P10), 注释流 (v1 B6)
  |
  v
评测 (P12): 语义性 / 不变性 / 可学习性 / 负对照 + 零样本似然 (v1 D1) + 疾病预测
```

### 5.3 与 v1 方案的映射

| v1 | 状态 | 说明 |
|---|---|---|
| A1 语义蒸馏 split-RVQ | 升级为 P1 | 新增频谱平衡目标、SemHiTok 接口、实测证据 |
| A2 尺度侧信道 | 并入 P2 | 非周期流携带绝对功率 |
| A3 多速率 motif | 并入 P5 | |
| A4 motif 词汇化 | 路线 1 升级为 P6，路线 2 并入 P5 | |
| A5 传感器无关 tokenizer | 保留，关联 P4 | 新增 OSF 两阶段掩码与 SleepFM-2 可穿戴迁移的证据 |
| B1 跨模态联合 depth | 与 P10 结合 | |
| B2 全夜层级生成 | 保留，P7 为轻量替代 | |
| B3 多尺度预测 | JEPA 项升级为 P8 | |
| B4 MoT | 保留 | |
| B5 any-to-any | 保留 | |
| B6 注释流 | 保留，与 P11 结合 | |
| B7 双向 / 读出层 | 并入 P12 | |
| B8 生成 + 对比 | 保留为备选 | SleepFM-2 用对比 + MAE 联合目标 [4] 是新的佐证 |
| C1 Rho-1 | 保留 | 与 P3 互补 |
| C2 DoReMi + MeCo | 扩展为 P4 | |
| D1 零样本似然 + PhysioBLIMP | 并入 P12 | |
| D2 SAE 词典 | 保留 | P7 的概念词典提供离散替代 |

### 5.4 实验设计更新

- **基线**（相对 v1 新增）：SleepFM-2（代码公开 [4]）、SleepLM（代码公开 [9]）、sleep2vec [8]、BandVQ [23]、NeuroRVQ [24]、FAME [35]；tokenizer 级对照：TFM-Tokenizer [21]、CodeBrain [22]、固定离散化（PiMT 式子频带 token [26]，以及 [34] 的结论提示固定方案是必要对照）。
- **数据**：NSRR 队列（需签署数据使用协议）；SSSM 事件标注 [19] 用于事件级评测；Stanford Sleep Bench 协议用于疾病预测 [6]。
- **协议**：被试级划分、留一队列；所有疾病结果附人口学基线与标签置换负对照 [14][44]；tokenizer 结果附无解码评测 [67]。
- **消融**：每个组件在 50M-150M 参数、约 5,000 夜上单独开关，3 个随机种子；报告"语义-重构"与"语义-NLL"两张帕累托图。

### 5.5 时间线（约 5-6 个月）

| 阶段 | 时长 | 内容 |
|---|---|---|
| 0 | 2-3 周 | 用已发布权重跑 P12 诊断套件（FMScope、频谱审计、DFA 恢复、purity）；P2 的离线非周期统计；在真实 PSG 上复现本文 probe；搭建 tokenizer 训练代码 |
| 1 | 4-6 周 | P1 + P2 tokenizer（P3 作为消融）；小规模语言模型验证 |
| 2 | 6-8 周 | P7 + P4，中等规模；P5 或 P6 二选一 |
| 3 | 4-6 周 | 全量训练、完整评测、写作 |

### 5.6 审稿风险与应对

| 可能的质疑 | 应对 |
|---|---|
| "语义 token"只是多任务监督 | 提供 HuBERT 式自举与频谱平衡两个无标注版本；P12 负对照；报告去掉描述子后的退化幅度 |
| 与 SleepFM-2 相比疾病预测不占优 | 明确差异点为 token 语义、生成与零样本能力、可解释概念词典；在相同协议下如实报告；保留 v1 B8 作为备选 |
| 合成实验不能代表真实 PSG | 阶段 0 在 NSRR 上复现 probe（用 YASA 检测的纺锤波 / 慢波作为事件标签） |
| 只是把语音 / LLM 技巧搬过来 | 每个组件对应一条有文献证据的失败模式（第 2 节），并给出生理动机 |
| 与 BandVQ / FAME / CodeBrain 的差异 | 它们是单模态、编码器式；本工作是多模态、生成式、离散 motif，且做了非周期 / 周期与主体因子化 |
| 数据泄漏与队列偏差 | 被试级划分、留一队列、人口学基线 |

---

## 6. 附录

### 6.1 `semantic_vq_probe.py` 的输出

`python docs/research/semantic_vq_probe.py`（numpy 2.4.6、scipy 1.17.1）：

```text
24 subjects x 400 s, 64-code codebooks, 3 seeds
class balance: {'none': 7458, 'kcomplex': 534, 'alpha': 554, 'sigma': 533, 'beta': 521}
tokenizer               NMI(code,class)  NMI(code,subject)  NMI(code,phase)  bal.acc
A. waveform MSE                   0.097              0.078            0.347    0.405
B. flat spectrum                  0.172              0.365            0.026    0.688
C. band descriptors               0.207              0.281            0.030    0.775

chance balanced accuracy = 0.200

per-class recall of the majority-vote decoder (test subjects):
tokenizer                   none  kcomplex     alpha     sigma      beta
A. waveform MSE            0.244     0.963     0.246     0.381     0.190
B. flat spectrum           0.573     0.898     0.700     0.632     0.636
C. band descriptors        0.645     0.924     0.766     0.826     0.716
```

说明：k-means 是均方重构目标下的最优离散化器，用它代表"重构驱动的码本"是保守的（真实 RVQ 有编码器，但目标同为波形误差）。合成信号只用于说明目标函数的性质，不代表真实 PSG 上的效应量。

### 6.2 评估后未进入主表的候选

| 候选 | 借鉴 | 结论 |
|---|---|---|
| 连续 motif + 扩散头 | MAR、VibeVoice、CALM | 与 v1 结论相同：失去离散似然与词表分析能力，不作为主线 |
| 完全 tokenizer-free（H-Net 端到端） | [78] | 并入 P5 作为高风险路线；H-Net 在生物序列上有数据效率证据，但在多模态生理信号上未见 |
| 纯 JEPA 替代 NTP | [36][37][38] | 与"生成式 + 似然"的定位冲突；作为 P8 辅助目标 |
| RL 调优的语义瓶颈（TimeSRL） | [98] | 依赖 LLM 的自然语言抽象，与离散 motif 路线不同；可作为后续的临床解释接口 |
| 量化器替换（FSQ / LFQ / SimVQ） | [89] | 工程选项，并入 P1 |

### 6.3 参考文献

**睡眠与生理信号基础模型**

1. J. F. Carter, L. Tarassenko. Next-Token Prediction Learns Generalisable Representations of Sleep Physiology (Hypnos). arXiv:2606.09605, 2026.
2. R. Thapa et al. SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals. ICML 2024. arXiv:2405.17766.
3. R. Thapa et al. A multimodal sleep foundation model for disease prediction. Nature Medicine, 2026. doi:10.1038/s41591-025-04133-4.
4. R. Thapa et al. Learning transferable human physiology from two million hours of sleep with SleepFM-2. arXiv:2609.06849, 2026. 代码：github.com/zou-group/sleepfm-v2-public.
5. Z. Shuai, Z. Xu, D. Yang, W. Wang, Y. Yang. OSF: On Pre-training and Scaling of Sleep Foundation Models. ICML 2026. arXiv:2603.00190.
6. Stanford Sleep Bench: Evaluating Polysomnography Pre-training Methods for Sleep Foundation Models. arXiv:2512.09591, 2025.
7. SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures. ICML 2026. arXiv:2602.07628.
8. Yuan et al. sleep2vec: Unified Cross-Modal Alignment for Heterogeneous Nocturnal Biosignals. ICLR 2026.（arXiv 编号待核对）
9. SleepLM: Natural-Language Intelligence for Human Sleep. ICML 2026 (spotlight). arXiv:2602.23605. 代码：github.com/yang-ai-lab/SleepLM.
10. K. Zha et al. Physiology as Language: Translating Respiration to Sleep EEG. ICML 2026. arXiv:2602.00526.
11. A unified time-frequency foundation model for sleep decoding (SleepGPT). Nature Communications, 2025. doi:10.1038/s41467-025-67970-4.
12. W. Coon, M. Ogg. Sleep EEG foundation models reveal within-stage microstructure that improves health screening beyond traditional stages. npj Digital Medicine, 2026.
13. B. Fox et al. A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages (PFTSleep). SLEEP, 2025.
14. A. Helmy, R. Morand, A. Calzoni et al. Foundation Models in Sleep Research: Opportunities and Limitations. SLEEP, 2026. doi:10.1093/sleep/zsag225.
15. D. R. Mazzotti, G. M. Travers, B. Fox, A. Parekh. Sleep Foundation Models: Present Performance, Future Potential. SLEEP, 2026. doi:10.1093/sleep/zsag228.
16. H. Mehdi et al. Comparative Analysis of State-of-the-Art Foundation Models for Sleep Analysis Under Channel Reduction. arXiv:2609.22105, 2026.
17. W. Lehn-Schiøler et al. Pretraining on Sleep Data Improves non-Sleep Biosignal Tasks. arXiv:2605.02500, 2026.
18. R. Thapa, U. Hanif, R. Guillard et al. Agentic AI-enabled discovery across large-scale sleep physiology. arXiv:2607.25175, 2026.
19. Semantic Segmentation of Sleep Events for High-Resolution Sleep Decoding (SSSM). medRxiv 2025.09.25.25336636.
20. R. B. Berry et al. Rules for scoring respiratory events in sleep: update of the 2007 AASM Manual. J. Clin. Sleep Med. 8(5), 2012; AASM Manual for the Scoring of Sleep and Associated Events.

**EEG / 生理 tokenizer 与预训练目标**

21. J. Pradeepkumar et al. Tokenizing Single-Channel EEG with Time-Frequency Motif Learning (TFM-Tokenizer). ICLR 2026. arXiv:2502.16060.
22. CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model. ICLR 2026. arXiv:2506.09110.
23. J. Sukhbaatar, S. Imamura, T. Tanaka. BandVQ: Band-Wise Vector-Quantized EEG Foundation Model. arXiv:2605.24921, 2026.
24. NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models. arXiv:2510.13068, 2025.
25. BrainRVQ: A High-Fidelity EEG Foundation Model via Dual-Domain Residual Quantization and Hierarchical Autoregression. arXiv:2602.16951, 2026.
26. Beyond Hearing: Learning Task-Agnostic ExG Representations from Earphones via Physiology-Informed Tokenization (PiMT). ICLR 2026. arXiv:2510.20853.
27. W. Jiang et al. NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals. ICLR 2025. arXiv:2409.00101.
28. W. Jiang et al. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (LaBraM). ICLR 2024.
29. BrainOmni. NeurIPS 2025. arXiv:2505.18185; REVE. NeurIPS 2025. arXiv:2510.21585.
30. J. Jin et al. Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model (HeartLang). ICLR 2025. arXiv:2502.10707.
31. On the role of the tokenizer in ECG transformer models. arXiv:2609.15433, 2026.
32. Beat-Synchronous Tokenization for ECG Transformers. IEEE MLSP 2026. arXiv:2608.30367.
33. ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling. arXiv:2412.14373.
34. A Systematic Evaluation of Sample-Level Tokenization Strategies for MEG Foundation Models. arXiv:2602.16626, 2026.
35. J. Yu et al. Understanding and Correcting Low-Frequency Bias in EEG Foundation Model (FAME). arXiv:2608.01898, 2026.
36. Laya: A LeJEPA Approach to EEG via Latent Prediction over Reconstruction. arXiv:2603.16281, 2026.
37. EEG-JEPA: Structured Latent Prediction for EEG Foundation Models. arXiv:2608.00114, 2026.
38. K. Yu et al. Neural State Prediction: Obstructing Shortcut Learning in EEG Foundation Models. arXiv:2609.31167, 2026.
39. Multimodal EEG World Model: Self-Supervised Latent Transition Learning for Wearable EEG Seizure Detection. medRxiv 2026.07.20.26358450.
40. Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders. arXiv:2605.13930, 2026.

**诊断、审计与基准**

41. Aperiodic and Low-Frequency Spectral Bias in Reconstruction based EEG Foundation Models. arXiv:2605.26434, 2026.
42. J. S. Bindra, S. Panwar, S. R. Chowdhury. A spectral audit framework reveals task-dependent aperiodic reliance across EEG and ECG deep learning. arXiv:2606.08583, 2026.
43. J.-Y. Lin et al. The Identity Trap in EEG Foundation Models: A Diagnostic Audit (FMScope). arXiv:2606.06647, 2026.
44. A Negative-Control Protocol for Clinical EEG Foundation-Model Benchmarks: Dataset Identity and External-Cohort Stress Testing. arXiv:2607.24519, 2026.
45. Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in EEG Foundation Models. arXiv:2606.09189, 2026.
46. M. Zare. Foundation Models for EEG Are Blind to Long-Range Temporal Correlations: A Spectral-Temporal Dissociation Behind Their Cross-Population Fragility. arXiv:2607.24834, 2026.
47. Benchmarking EEG Foundation Models at Scale: Lessons from 20,000 Evaluations (EEG-Arena). arXiv:2609.32743, 2026.
48. EEG-FM-Compass: Progress, Benchmarking, and Future Directions for EEG Foundation Models. arXiv:2601.17883; National Science Review, 2026.
49. Brain4FMs. arXiv:2602.11558; OmniEEG-Bench. arXiv:2606.00815, 2026.

**语音与音频**

50. A. Défossez et al. Moshi: a speech-text foundation model for real-time dialogue (Mimi). arXiv:2410.00037.
51. X. Zhang et al. SpeechTokenizer. ICLR 2024. arXiv:2308.16692.
52. Z. Ye et al. X-Codec. AAAI 2025. arXiv:2408.17175; X-Codec2 (Llasa). arXiv:2502.04128.
53. XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs. ACL 2026. arXiv:2506.23325.
54. BiMTokenizer: Preserving Semantic-Acoustic Balance in Low-Bitrate Speech Tokenization via Bidirectional State-Space Modeling. EMNLP 2026. arXiv:2609.00562.
55. ReLMCodec: Designing Predictable Speech Tokens from Pre-Quantization Phoneme Structure. arXiv:2608.08286, 2026.
56. H.-L. Chung, Y. Chen, H.-y. Lee. LLM-Codec: Neural Audio Codec Meets Language Model Objectives. Findings of ACL 2026. arXiv:2604.17852.
57. Z. Huang et al. Kanade: A Simple Disentangled Tokenizer for Spoken Language Modeling. arXiv:2602.00594, 2026.
58. Z. Ju et al. NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models (FACodec). ICML 2024. arXiv:2403.03100.
59. Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding (VARSTok). AAAI 2026. arXiv:2509.04685.
60. Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization (DyCAST). arXiv:2601.23174, 2026.
61. Interpreting and Evaluating Dynamic-Rate Speech Codec Boundaries. arXiv:2609.36951, 2026.
62. FlexiCodec. arXiv:2510.00981; DTM-Codec. arXiv:2606.29480.
63. A. Baade et al. SyllableLM. ICLR 2025. arXiv:2410.04029; Sylber. arXiv:2410.07168.
64. UniAudio-Token: Empowering Semantic Speech Tokenizers with General Audio Perception. arXiv:2605.31521, 2026.
65. TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling. arXiv:2508.16790.
66. W.-N. Hsu et al. HuBERT. IEEE/ACM TASLP 2021; C.-C. Chiu et al. BEST-RQ. ICML 2022.
67. DASB: Discrete Audio and Speech Benchmark. arXiv:2406.14294; Discrete Audio Tokens: More Than a Survey! arXiv:2506.10274.
68. S. Cuervo, R. Marxer. Scaling Properties of Speech Language Models. EMNLP 2024.
69. T. A. Nguyen et al. The Zero Resource Speech Benchmark 2021. arXiv:2011.11588.
70. F. Shen et al. Acoustic BPE for Speech Generation with Discrete Tokens. ICASSP 2024.

**文本与大语言模型**

71. G. Bachmann, V. Nagarajan. The Pitfalls of Next-Token Prediction. ICML 2024. arXiv:2403.06963.
72. F. Gloeckle et al. Better & Faster Large Language Models via Multi-token Prediction. ICML 2024.
73. Y. Liu et al. Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models (ConceptLM). arXiv:2602.08984, 2026.
74. NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction. arXiv:2609.10715, 2026.
75. C. Shi et al. Hierarchical Latent Prediction for Language Models (HiLP). arXiv:2608.05806, 2026.
76. LCM team. Large Concept Models: Language Modeling in a Sentence Representation Space. arXiv:2412.08821.
77. A. Pagnoni et al. Byte Latent Transformer. ACL 2025. arXiv:2412.09871.
78. S. Hwang, B. Wang, A. Gu. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net). ICLR 2026. arXiv:2507.07955; H-Net++. arXiv:2508.05628.
79. SuperBPE: Space Travel for Language Models. arXiv:2503.13423.
80. Z. Lin et al. Not All Tokens Are What You Need for Pretraining (Rho-1). NeurIPS 2024.
81. S. Rajput et al. Recommender Systems with Generative Retrieval (TIGER). NeurIPS 2023. arXiv:2305.05065.

**视觉**

82. J. Yao et al. Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models (VA-VAE). CVPR 2025. arXiv:2501.01423.
83. T. Xiong et al. GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation. ICCV 2025.
84. Z. Chen et al. SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook. arXiv:2503.06764.
85. UniTok. arXiv:2502.20321; TokLIP. arXiv:2505.05422.
86. X. Wen et al. "Principal Components" Enable A New Language of Images (Semanticist). ICCV 2025. arXiv:2503.08685.
87. Masked Autoencoders Are Effective Tokenizers for Diffusion Models (MAETok). ICML 2025; Latent Denoising Makes Good Tokenizers (l-DeTok). arXiv:2507.15856.
88. Studying Image Tokenizers as Visual Languages in Unified Multimodal Models. arXiv:2609.09143, 2026.
89. Y. Zhu et al. Addressing Representation Collapse in Vector Quantized Models with One Linear Layer (SimVQ). ICCV 2025. arXiv:2411.02038; F. Mentzer et al. FSQ. ICLR 2024; L. Yu et al. Language Model Beats Diffusion (LFQ). ICLR 2024.

**时间序列**

90. Time Series as Language: A Universal Tokenizer for General-Purpose Time Series Foundation Models (UniTok-FM). arXiv:2606.09861, 2026.
91. L. Götz et al. Byte Pair Encoding for Efficient Time Series Forecasting. arXiv:2505.14411（检索摘要称 ICML 2026）.
92. Small Vocabularies, Big Gains: Pretraining and Tokenization in Time Series Models. arXiv:2511.11622.
93. EIDOS: Latent-Space Predictive Learning for Time Series Foundation Models. arXiv:2602.14024, 2026.
94. LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series Representation Learning. KDD MILETS 2026. arXiv:2607.00958.
95. One-Step Next-Latent Prediction Is Not a World Model. arXiv:2609.36227, 2026.
96. A. F. Ansari et al. Chronos. TMLR 2024.
97. Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization. ICML 2025. arXiv:2412.05244.
98. TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs. arXiv:2605.21295; ACM IMWUT 2026.

**似然与分布外检测**

99. J. Ren et al. Likelihood Ratios for Out-of-Distribution Detection. NeurIPS 2019.
100. J. Serrà et al. Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models. ICLR 2020.
