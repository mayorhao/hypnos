# Agent 指令：逐秒语义描述子的免训练快速验证（T0-T5）

> 用途：把本文件整体交给执行 agent。它规定描述子的精确定义、每个实验的设置、预注册的通过标准和输出格式。
> 背景：实验 1.2 发现 Hypnos 的 8 级 RVQ token（`eeg-q8-causal`，码本 2048，约 88 bit/s）在 alpha 及以上频段丢失信息，纺锤波事件级 F1 只有 0.76。计划在 split RVQ 中加入 HuBERT 式语义层：第一轮用逐秒描述子做 k-means 得到伪标签，再让语义分支对齐这些目标。本任务在不训练任何神经网络的前提下，判断这个方向是否值得推进。
> 版本：2026-09-30

---

## 0. 任务与边界

**要回答的五个问题**

| 编号 | 问题 |
|---|---|
| T1 | 描述子聚出来的码是否对应生理事件，而不是主体身份 |
| T2 | Hypnos 的 8 级 token 是否缺少这部分信息（核心） |
| T3 | 冻结的 tokenizer 编码器能否产生这些语义码 |
| T4 | 语义码对夜级结局（年龄、BMI、AHI）是否有增量 |
| T5 | 把一级 RVQ 让给语义层要付出多少码率代价 |

**允许**：k-means、PCA、岭回归、逻辑回归、计数 n-gram、YASA 检测、已发布 tokenizer 与 Hypnos 的推理。

**禁止**：训练或微调任何神经网络，包括 tokenizer 与 Hypnos。

**开始前**：先阅读实验 1.2 的脚本 `experiments/2026-09-30_recon_fidelity/followup.py`（7.2 节的 z 与 token 特征提取、受试者折划分）和 `redetect.py`（µV 换算、YASA 参数），尽量复用其中的函数与约定。与本文冲突时以本文为准，并把差异写进 README。

**运行顺序**：T0 → T0 自检 → 写 `prereg.json` → T1 → T2 → T3 → T5 → T4。T4 放最后，因为它可能需要扩充夜数。

**预计时间**：T0 1 天，T1 0.5 天，T2 1 天，T3 0.5 天，T5 0.5 天，T4 1 天。

---

## 1. 输入与输出位置

| 内容 | 位置 |
|---|---|
| 实验 1.2 记录与脚本 | `experiments/2026-09-30_recon_fidelity/`（README、`report.json`、`subjects.txt`、`followup.py`、`redetect.py`） |
| 实验 1.2 大文件 | `/data/jfan/hypno/runs/2026-09-30_recon_fidelity/`（`recon/`、`metrics/`、`report_followup.json`、`report_redetect.json`） |
| 受试者列表 | `subjects.txt`：SHHS1 测试集（OSF `patient_pretrain_test`）200 夜，每人一夜 |
| tokenizer | Hypnos `eeg-q8-causal`：128 Hz，每秒 1 个 token，8 级 RVQ，每级 2048 码 |
| 预处理 | 因果 0.5-45 Hz 带通 + 陷波 + 滚动 z-score（τ = 60 s）+ 对数压缩（拐点 8σ）；每秒尺度 `log_scale` |
| YASA | conda 环境 `yasa`，YASA 0.7.0；使用第 8 节在原始信号上自检测的纺锤波与慢波 |
| NSRR 标注 | 分期与觉醒（与 7.4 节相同来源） |
| 夜级变量 | SHHS1 数据集 CSV：年龄、性别、BMI、AHI。字段名以 SHHS1 数据字典为准，候选为 `age_s1`、`gender`、`bmi_s1`、`ahi_a0h3`，使用前必须核对 |
| 新实验目录 | `experiments/2026-10-XX_semantic_descriptor_quicktest/`（脚本、README、`prereg.json`、各报告） |
| 新大文件目录 | `/data/jfan/hypno/runs/2026-10-XX_semantic_descriptor_quicktest/`（逐夜描述子、码、预测） |

路径若与实际不符，先确认再运行，并在 README 中记录实际路径。

---

## 2. 通用约定

**时间索引**：token t 覆盖预处理信号的样本 `[t*128, (t+1)*128)`。`z_t` 是该 token 的量化前编码器输出；描述子第 t 行只使用到 `(t+1)*128` 为止的样本（纺锤波形态特征除外，见 3.2）。

**对齐自检**：用本文的 z 特征复现 7.2 节"连续 z 预测 alpha 对数功率"的 R²（报告值约 0.89）。偏差超过 0.02 说明时间对齐有误，必须先修正。

**有效秒**：沿用实验 1.2 的质量掩码（质量合格、有分期的 30 s epoch 内的秒），再去掉描述子含 NaN 的秒（每夜开头 3 s）。所有拟合与评测只用有效秒。

**折**：受试者级 5 折。优先复用 7.2 节的折划分；没有保存时，用种子 0 随机生成，并写入 `folds.json`。所有需要拟合的步骤，包括 k-means、全局标准化、PCA、岭回归、逻辑回归、n-gram、超参数选择，都只能在训练折上完成。每夜内的稳健标准化只用该夜自身的数据，不构成泄漏。

**通道**：C3 为主结果；T1 与 T2 的主要指标在 C4 上重复一遍，只要求方向一致。

**随机性与置信区间**：k-means 用种子 0、1、2，报告均值与标准差。所有差值的 95% 置信区间用受试者级 bootstrap（1,000 次，重抽测试受试者，使用折外预测）。

**评测集**：每个任务在测试折中，从该任务掩码内的有效秒里均匀随机抽 20%（种子 0），保持自然患病率。所有特征组使用同一评测集。

**训练集**：每个任务在训练折中，取全部阳性秒，再随机抽取至多 20 倍数量的阴性秒（种子 0）。所有特征组使用同一训练集。

---

## 3. T0：描述子、离散化与标签

### 3.1 µV 域信号

描述子在 µV 域计算，不在滚动 z-score 后的归一化域计算，因为滚动归一化会抹掉整夜的功率变化。换算方式与第 8 节相同：

```text
x_uV[n] = x_proc[n] * exp(log_scale[t]),  t = floor(n / 128)
```

压缩拐点在 8σ，这一换算在拐点以下是精确的。

### 3.2 描述子定义

每个通道每秒一个 10 维向量。前 7 维称为 `D_band`，全部 10 维称为 `D_full`。

| 维 | 名称 | 定义 | 窗口 |
|---|---|---|---|
| 1 | so | 0.5-1.25 Hz 功率，log10 µV² | 4 s 尾随窗，多窗谱 NW = 2，3 个 DPSS 窗 |
| 2 | delta | 1.25-4 Hz | 同上 |
| 3 | theta | 4-8 Hz | 同上 |
| 4 | alpha | 8-11 Hz | 2 s 尾随窗，多窗谱 NW = 1.5，2 个 DPSS 窗 |
| 5 | sigma_slow | 11-13 Hz | 同上 |
| 6 | sigma_fast | 13-16 Hz | 同上 |
| 7 | beta | 16-30 Hz | 同上 |
| 8 | rel_sigma | 11-16 Hz 功率与 1-30 Hz 功率之比（线性） | 2 s 尾随窗，同上 |
| 9 | sigma_corr_max | 秒内各 0.3 s 子窗（步长 0.1 s）上，11-16 Hz 滤波信号与 1-30 Hz 滤波信号的 Pearson 相关，取最大值 | 子窗终点落在该秒内 |
| 10 | sigma_rms_max | 同样的子窗上，11-16 Hz 滤波信号 RMS 的 log10，取最大值 | 同上 |

说明：

- 频段划分与实验 1.2 一致（alpha 8-11 Hz、sigma 11-16 Hz），sigma 再分为慢与快两段；0.5-1.25 Hz 单列，是 Sleep2.0 区分 N2 亚状态时发现差异的频段。
- 低频用 4 s 窗，是因为 2 s 窗的频率分辨率不足以估计慢振荡与 delta。
- 第 9、10 维的定义与 YASA 纺锤波检测判据一致。它们必须用零相位滤波（`sosfiltfilt`）：若对两个频带分别使用因果滤波，两者在 13 Hz 处的相位延迟不同，会系统性压低相关系数（参考实现的测试中，纺锤波期间相关从 0.97 降到 0.46）。零相位滤波带来约数百毫秒的前瞻。描述子只作训练目标，推理时不需要，这一前瞻可以接受，在 README 中注明即可。
- **不放入描述子**：`log_scale`（绝对幅度）与非周期偏移。它们属于非周期流或每夜身份信息，不属于语义层。

**参考实现**（已在 8 小时合成信号上测试：单夜约 3 s；纺锤波所在秒的 sigma_corr_max 为 0.97，1/f 背景下中位数约 0.48）：

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import dpss

FS = 128
LOW_BANDS = {"so": (0.5, 1.25), "delta": (1.25, 4.0), "theta": (4.0, 8.0)}
HIGH_BANDS = {"alpha": (8.0, 11.0), "sigma_slow": (11.0, 13.0), "sigma_fast": (13.0, 16.0), "beta": (16.0, 30.0)}


def mt_band_logpow(x, n_seconds, win_s, nw, bands, fs=FS, chunk=2048):
    """Causal trailing-window multitaper band power (log10, uV^2). Row t uses samples [(t+1)*fs - win, (t+1)*fs)."""
    win = int(win_s * fs)
    tapers = dpss(win, nw, Kmax=int(2 * nw - 1))  # unit-energy tapers, shape (K, win)
    freqs = np.fft.rfftfreq(win, 1 / fs)
    df = freqs[1] - freqs[0]
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands.values()]
    out = np.full((n_seconds, len(bands)), np.nan, np.float32)
    x = np.asarray(x[: n_seconds * fs], dtype=np.float64)
    first = int(np.ceil(win / fs)) - 1
    if n_seconds <= first:
        return out
    starts = np.arange(first, n_seconds) * fs + fs - win
    for i in range(0, len(starts), chunk):
        s = starts[i : i + chunk]
        seg = x[s[:, None] + np.arange(win)[None, :]]
        seg = seg - seg.mean(1, keepdims=True)
        spec = np.abs(np.fft.rfft(seg[:, None, :] * tapers[None], axis=-1)) ** 2
        psd = spec.mean(1) / fs
        psd[:, 1:-1] *= 2.0  # one-sided
        for j, m in enumerate(masks):
            out[first + i : first + i + len(s), j] = np.log10(psd[:, m].sum(1) * df + 1e-12)
    return out


def spindle_shape(x, n_seconds, fs=FS, win_s=0.3, step_s=0.1):
    """Per-second max over 0.3 s sub-windows of sigma/broadband correlation and log10 sigma RMS.

    Zero-phase filters (as YASA): separate causal filters would introduce a band-dependent
    phase lag and bias the correlation downwards.
    """
    x = np.asarray(x[: n_seconds * fs], dtype=np.float64)
    xs = sosfiltfilt(butter(4, [11.0, 16.0], "bandpass", fs=fs, output="sos"), x)
    xb = sosfiltfilt(butter(4, [1.0, 30.0], "bandpass", fs=fs, output="sos"), x)
    w, h = int(win_s * fs), int(step_s * fs)
    ws = sliding_window_view(xs, w)[::h]
    wb = sliding_window_view(xb, w)[::h]
    ends = np.arange(ws.shape[0]) * h + w  # exclusive end sample of each sub-window
    sec = (ends - 1) // fs
    a = ws - ws.mean(1, keepdims=True)
    b = wb - wb.mean(1, keepdims=True)
    corr = (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)
    rms = np.log10(np.sqrt((ws**2).mean(1)) + 1e-12)
    corr_max = np.full(n_seconds, np.nan, np.float32)
    rms_max = np.full(n_seconds, np.nan, np.float32)
    np.fmax.at(corr_max, sec, corr.astype(np.float32))
    np.fmax.at(rms_max, sec, rms.astype(np.float32))
    return corr_max, rms_max


def descriptors(x_uv, n_seconds, fs=FS):
    """D_band (n, 7) and D_full (n, 10). Columns: so, delta, theta, alpha, sigma_slow, sigma_fast, beta,
    rel_sigma, sigma_corr_max, sigma_rms_max."""
    low = mt_band_logpow(x_uv, n_seconds, 4.0, 2.0, LOW_BANDS, fs)
    high = mt_band_logpow(x_uv, n_seconds, 2.0, 1.5, HIGH_BANDS, fs)
    rel = mt_band_logpow(x_uv, n_seconds, 2.0, 1.5, {"sigma": (11.0, 16.0), "total": (1.0, 30.0)}, fs)
    rel_sigma = 10 ** (rel[:, 0] - rel[:, 1])
    corr_max, rms_max = spindle_shape(x_uv, n_seconds, fs)
    d_band = np.concatenate([low, high], 1)
    d_full = np.concatenate([d_band, rel_sigma[:, None], corr_max[:, None], rms_max[:, None]], 1)
    return d_band.astype(np.float32), d_full.astype(np.float32)


def per_night_robust_z(D, valid, clip=5.0):
    """Per-night, per-dimension (d - median) / IQR on valid seconds, clipped to [-clip, clip]."""
    med = np.nanmedian(D[valid], 0)
    q75, q25 = np.nanpercentile(D[valid], [75, 25], 0)
    return np.clip((D - med) / (q75 - q25 + 1e-6), -clip, clip).astype(np.float32)
```

### 3.3 归一化

1. 每夜、每维：用该夜有效秒的中位数与四分位距做稳健标准化，截断到 ±5（`per_night_robust_z`）。
2. 全局、每维：在训练折有效秒上计算均值与标准差，再标准化一次，使 k-means 中各维权重相同。

### 3.4 对照描述子

| 名称 | 定义 | 用途 |
|---|---|---|
| `D_raw` | 4 s 尾随窗多窗谱的 log10 功率，在 1-30 Hz 内按 1 Hz 取平均（29 维），只做全局逐维标准化，不做每夜标准化 | 近似"直接用 P(f) 向量"的做法 |
| `D_epoch30`（可选） | 每个 30 s epoch 的 log10 Welch 功率谱（4 s Hann 窗），0.5-30 Hz 按 1 Hz 取平均，赋给该 epoch 的每一秒，全局标准化 | 近似 Sleep2.0 第一轮的时间分辨率 |
| `D_full+fast`（消融） | `D_full` 与"`D_full` 减去其 60 s 尾随滑动中位数"拼接，共 20 维 | 检验快慢分解能否避免聚成分期码 |

### 3.5 离散化

所有离散化使用相同的 K。主结果 K = 128，敏感性分析 K = 64 与 256。

| 名称 | 聚类对象 | 说明 |
|---|---|---|
| `S_full` | 标准化后的 `D_full` | 主方案 |
| `S_band` | 标准化后的 `D_band` | 纺锤波任务的主结果，避免循环论证 |
| `S_raw` | `D_raw` | 对照 |
| `S_epoch30`（可选） | `D_epoch30` | 对照 |
| `Z_K` | 量化前连续 z（单秒，256 维，不标准化） | 对照：编码器已有的结构 |
| `E_K` | token 嵌入 e_t = 8 级码字向量之和（256 维） | 对照：量化后的结构 |
| `L1_K` | 第 1 级 2048 个码字向量，以训练折使用次数为权重做 k-means 分成 K 组；token 按其第 1 级码字映射到组 | 对照：RVQ 第一级的粗粒度版本 |

k-means 设置：`sklearn.cluster.MiniBatchKMeans(n_clusters=K, batch_size=10000, n_init=3, max_iter=300, random_state=seed)`。每个训练夜至多抽 3,000 个有效秒参与拟合，避免长夜主导。`L1_K` 用 `KMeans` 加 `sample_weight`。

### 3.6 标签

| 标签 | 定义 | 任务掩码 |
|---|---|---|
| 分期 | 30 s epoch 标签广播到每秒，W、N1、N2、N3、REM；未评分秒剔除 | 全部有效秒 |
| 觉醒 | NSRR 觉醒事件（与 7.4 节同一来源，RERA 不计入），与 `[t, t+1)` 有重叠即为阳性 | N1、N2、N3、REM 秒 |
| 纺锤波 | 第 8 节在原始信号上自检测的 YASA 纺锤波，与 `[t, t+1)` 重叠至少 0.25 s 即为阳性 | N2 秒 |
| 慢波 | 第 8 节自检测的 YASA 慢波，负峰落在 `[t, t+1)` 即为阳性 | N2 与 N3 秒 |
| 主体 | 受试者 ID | 全部有效秒 |

### 3.7 T0 自检（必须全部通过才进入 T1）

1. **对齐**：见第 2 节。
2. **生理方向**：按分期计算每维描述子的中位数，应满足：
   - so、delta 在 N3 最高；
   - sigma_slow、sigma_fast、rel_sigma 在 N2 高于 N3 和 REM；
   - alpha 在 W 最高；
   - beta 在 W 高于 N2 和 N3。
3. **纺锤波判别**：在 N2 秒中，YASA 纺锤波阳性秒与阴性秒的 sigma_corr_max、sigma_rms_max 的 AUROC 均应大于 0.8。

任何一项不通过，先排查换算、滤波或对齐问题，不进入 T1。

---

## 4. 预注册

在运行 T1 之前写入 `prereg.json`，之后不得修改。若事后确需增加分析，只能在报告中另列为"事后分析"。默认内容如下，可在运行前调整数值：

```json
{
  "primary_channel": "C3",
  "K_primary": 128,
  "T1": {
    "T1a": "arousal AUPRC: S_full minus max(E_K, L1_K) > 0, 95% CI excludes 0",
    "T1b": "N2 spindle AUPRC: S_band minus max(E_K, L1_K) > 0, 95% CI excludes 0",
    "T1c": "S_full: NMI(code, subject) <= 0.5 * S_raw AND split-half fingerprint top-1 <= 0.5 * S_raw",
    "pass": "(T1a OR T1b) AND T1c"
  },
  "T2": {
    "primary_semantic": "S_band",
    "pass": "for arousal OR N2 spindle: delta AUPRC (A+S minus A) >= 0.03 AND >= 10% of A, 95% CI excludes 0; C4 same sign"
  },
  "T3": {"pass": "retention = delta(A+S_hat) / delta(A+S) >= 0.7 on every task that passed T2"},
  "T4": {
    "min_nights_for_decision": 1000,
    "pass": "delta R2 (M1 minus M0) > 0 with 95% CI excluding 0 for age OR log(1+AHI)"
  },
  "T5": {"pass": "predicted delta-band MSC under the 1+7 configuration >= 0.95 after calibration"}
}
```

---

## 5. T1：语义性

**目的**：检验描述子码是否对应事件、分期，同时不编码主体身份。

**对象**：3.5 节全部离散化 × K ∈ {64, 128, 256} × 种子 {0, 1, 2}。

**指标**（均在测试折评测集上计算）：

1. **分期一致性**：码与分期的 AMI。
2. **事件可检测性**：只用第 t 秒码的独热编码（不加滞后），逻辑回归检测觉醒、纺锤波、慢波，报告 AUPRC 与 AUROC。模型为 `LogisticRegression(penalty="l2", solver="saga", max_iter=300, tol=1e-3)`，C 在 {0.01, 0.1, 1} 中用训练折内的受试者 3 折选择（以 AUPRC 为准）。
3. **主体信息**：码与主体 ID 的 NMI，每折单独计算后取平均。
4. **半夜指纹匹配**：对每夜，把睡眠期（首个到最后一个睡眠 epoch）按时间分成前后两半，分别计算码占比直方图（使用该夜作为测试夜时所在折的 k-means 模型）；取平方根后用余弦相似度（Hellinger），用每个前半夜在全部 200 个后半夜中检索，报告 top-1 准确率。随机水平为 0.005。
5. **游程**：码的平均游程长度（秒），总体与分期分别报告。游程很长说明码退化成了分期标签。

**输出**：`report_T1.json`，以及 K = 128 的汇总表：每种离散化一行，列为上述指标（种子均值 ± 标准差）；再加三个差值及其置信区间：S_full 对 max(E_K, L1_K) 的觉醒 AUPRC 差、S_band 对 max(E_K, L1_K) 的纺锤波 AUPRC 差、S_full 对 S_raw 的主体 NMI 与指纹比值。

**判定**：按 `prereg.json` 的 T1 条款。

---

## 6. T2：信息缺口（核心）

**目的**：在 Hypnos 语言模型实际看到的 token 之外，语义码能否提供额外的事件信息。

**特征**（每秒 t 使用 t-3 到 t 共 4 秒）：

| 名称 | 构成 | 维度 |
|---|---|---|
| A | 8 级码字逐级独热，按滞后分开 | 4 × 8 × 2048 = 65,536，稀疏 |
| S | `S_band` 码独热（主结果）；`S_full` 作次要结果 | 4 × K |
| A+S | 拼接 | |
| Z | 连续 z，按训练折标准化 | 4 × 256 = 1,024 |
| Z+S | 拼接 | |
| D（参考） | 标准化 `D_band`，4 秒 | 28 |

A 必须用逐级独热，不能用码字向量之和。逐级独热等价于语言模型第一层看到的"逐级可学习嵌入之和"；码字向量之和会混入"线性不可读"的问题。

**任务**：觉醒、N2 纺锤波、N2 与 N3 慢波（二分类，报告 AUPRC 与 AUROC）；分期五分类（报告 macro-F1，作为健全性检查）。

**模型**：与 T1 相同的逻辑回归与 C 的选择方式；分期用多项式逻辑回归。65,536 维稀疏特征若 `saga` 太慢，可对全部特征组统一改用 `SGDClassifier(loss="log_loss")`，alpha 在 {1e-6, 1e-5, 1e-4} 中选择。所有特征组必须使用同一模型族。

**指标**：

- ΔAUPRC = AUPRC(A+S) − AUPRC(A)，含受试者 bootstrap 置信区间；
- 缺口闭合比例 = ΔAUPRC / (AUPRC(Z) − AUPRC(A))，仅当分母大于 0.01 时报告；
- 同时报告 Z+S 相对 Z 的提升，用于判断语义码相对编码器是否也有增量。

**禁止**：不要用任何频带功率作 T2 的目标。描述子本身就含频带功率，会构成循环论证。

**输出**：`report_T2.json`，以及 C3 主表（任务 × 特征组）和 C4 的 ΔAUPRC 表。

**判定**：按 `prereg.json` 的 T2 条款。

---

## 7. T3：可获得性（模拟语义层）

**目的**：判断在不重训编码器的情况下，从量化前的 z 能否得到与描述子码一致的语义码。

**步骤**：

1. 用岭回归从 Z 特征（z_{t-3..t}，1,024 维，标准化）预测标准化后的 `D_band`（7 维）与 `D_full`（10 维）。alpha 在 {0.1, 1, 10, 100, 1000} 中用训练折内受试者 3 折选择。报告每一维在评测集上的 R²。
2. 模拟语义码：把预测值 d̂_t 映射到该折 `S_band`（K = 128）k-means 的最近中心，得到 ĉ_t。注意 d̂_t 与中心必须处于同一标准化空间。
3. 一致性：ĉ_t 与 c_t 的准确率与 AMI。
4. 用 ĉ_t 代替 c_t 重做 T2（特征 A+Ŝ），计算保留率 = ΔAUPRC(A+Ŝ) / ΔAUPRC(A+S)，只对 T2 通过的任务计算。
5. 对照：把第 1 步的回归输入换成 A（逐级独热，稀疏岭回归），得到 ĉ_tok 与对应的保留率。

**解读**：

- 若 ĉ 的保留率高，现有编码器已经含有语义信息，只需加语义分支。
- 若 ĉ_tok 的保留率也高，说明语言模型第一层就能从 token 中线性算出这部分信息，此时 T2 的提升应当很小，两个结果应互相印证；不一致时需在报告中讨论。

**输出**：`report_T3.json`。

**判定**：按 `prereg.json` 的 T3 条款。

---

## 8. T5：码率代价

**目的**：估计把一级 RVQ 让给语义层的代价，并检查语义码序列是否便于语言模型预测。

**步骤**：

1. **有效码率**：用全部 200 夜的码字使用次数计算每级熵 H_k（bit），有效码率 R_eff = Σ H_k（bit/s），与名义值 88 bit/s 对比。
2. **语义码的可预测性**：对 `S_band`、`L1_K`、`Z_K`（K = 128）的逐秒序列，在训练折夜上训练加 0.1 平滑的二元与三元语法，在测试折夜上计算交叉熵 H_cond。只在连续有效秒段内计数，段间断开。报告 H(码)、H_cond 以及 H_cond / log2 K。
3. **游程**：引用 T1 的结果。
4. **注水预测**：
   - 在训练折夜的干净 N2 秒上，计算预处理后归一化域信号（tokenizer 实际看到的信号）的平均单边功率谱：Welch，4 s Hann 窗，0.25 Hz 分辨率，0-64 Hz。
   - 对三个码率求解水位 θ 并预测各频段 MSC：R_eff（校准）；R_ac(1+7) = Σ_{k=1..7} H_k（语义层替换第 8 级）；R_ac(1+8) = R_eff（语义层额外增加一级）。
   - 以 R_eff 下的预测与实验 1.2 在 7.1 节实测的 MSC 之差作为校准偏移，再报告 1+7 配置下 delta、theta、alpha、sigma 的校准后预测 MSC。
   - split RVQ 中语义码同样参与解码，所以只按声学码率计算是保守估计，需在报告中注明。

```python
import numpy as np


def waterfill_msc(f, S, rate_bits_per_s, bands):
    """Gaussian reverse water-filling. f: one-sided freq grid (Hz), S: one-sided PSD on f.
    Returns predicted power-weighted MSC per band for an MMSE coder at the given rate."""
    df = f[1] - f[0]
    S = np.maximum(S, S.max() * 1e-12)
    lo, hi = S.min() * 1e-6, S.max()
    for _ in range(200):  # bisection on the water level theta (log scale)
        th = np.sqrt(lo * hi)
        r = np.sum(np.maximum(0.0, np.log2(S / th))) * df
        lo, hi = (th, hi) if r > rate_bits_per_s else (lo, th)
    D = np.minimum(th, S)
    return {name: 1.0 - D[(f >= a) & (f < b)].sum() / S[(f >= a) & (f < b)].sum() for name, (a, b) in bands.items()}
```

**输出**：`report_T5.json`。

**判定**：按 `prereg.json` 的 T5 条款。不通过时，后续设计采用 1+8 配置。

---

## 9. T4：夜级下游收益

**目的**：检验语义码的占比特征，在已有夜级特征和 Hypnos 嵌入之外，对年龄、BMI、AHI 是否有增量。

**夜数**：先用现有 200 夜，结果只作方向参考。若要用于判定，需要从同一 OSF 划分中扩充到至少 1,000 夜。扩充只涉及预处理、tokenizer 编码和 Hypnos 推理，不涉及训练；k-means 模型仍只用训练折拟合。

**目标**：年龄、BMI、log(1 + AHI)。

**特征块**（每夜，只用有效秒与睡眠期）：

| 块 | 内容 |
|---|---|
| B0 协变量 | 年龄、性别、BMI、AHI，去掉当前目标本身 |
| B1 分期汇总 | 总睡眠时间、睡眠效率、入睡后清醒时间、入睡潜伏期、REM 潜伏期、各期占比（5）、每小时分期转换次数、各期平均持续时长（5） |
| B2 频谱汇总 | 整夜与 NREM 的 7 个频带平均对数功率和相对功率；NREM 平均谱在 2-30 Hz 上拟合的非周期指数 |
| B3 Hypnos 嵌入 | eeg_c3 的 1 Hz 嵌入在睡眠期与 N2 内的均值，拼接后 PCA 到 32 维（训练折拟合） |
| S_occ | `S_band`（K = 128）码在睡眠期的占比，加 N2 内的条件占比；取平方根后 PCA 到 32 维（训练折拟合） |

**模型**：岭回归，alpha 用内层受试者 5 折选择，外层使用第 2 节的 5 折。

**比较**：

- M0 = B0 + B1 + B2 + B3；M1 = M0 + S_occ（主比较）；
- M0a = B0 + B1 + B2；M1a = M0a + S_occ（与 Sleep2.0 的比较框架对应，其基线不含基础模型嵌入）。

**指标**：折外 R²；ΔR² 及受试者 bootstrap 置信区间；残差分析：用只含 S_occ 的模型的折外预测，与 M0 的折外残差计算 Spearman 相关。

**输出**：`report_T4.json`，注明实际夜数。夜数少于 1,000 时，结论栏只写"方向性"。

**判定**：按 `prereg.json` 的 T4 条款。

---

## 10. 汇总与判定

在实验目录写 `summary.md`，包含：

1. 每个测试的通过或不通过、关键数值与置信区间（C3；T1 与 T2 附 C4 方向）；
2. 下表的判定结果；
3. 与预注册的所有偏差。

| T1 | T2 | T3 | 结论 |
|---|---|---|---|
| 通过 | 通过 | 通过 | 推进。在现有编码器上加语义分支即可 |
| 通过 | 通过 | 不通过 | 推进，但需要重训编码器 |
| 通过 | 不通过 | 任意 | 语义层没有增量；信息已在 token 中，问题是可读性。转向语言模型侧读出或频带加权的重构损失 |
| 不通过 | 任意 | 任意 | 先修改描述子，不进入 split RVQ |

T4 决定论文的主张：通过时可主张下游收益；不通过时主张落在事件级任务、生成和可解释性上。T5 决定码率配置：1+7 还是 1+8。

---

## 11. 注意事项

1. **循环论证**：`D_full` 的第 8 至 10 维与 YASA 纺锤波判据相同。纺锤波任务一律以 `S_band` 为主结果，`S_full` 只作上限；觉醒与分期为专家标注，是最干净的判据。条件允许时，另用 MASS 或 DREAMS 的专家纺锤波标注复核。
2. **相同 K**：所有离散化之间的比较必须在同一 K 下进行；NMI、AMI 与 AUPRC 都随 K 变化。
3. **泄漏**：k-means、全局标准化、PCA、回归与超参数只在训练折拟合；评测集只来自测试折。
4. **不得事后调阈值**：见第 4 节。
5. **记录**：在 README 中写明软件版本（numpy、scipy、sklearn、YASA）、运行时间、实际路径、随机种子，以及与本文的任何偏差。
6. **失败也要报告**：任何测试不通过时，照常输出完整数值，不要删减或替换指标。
