# 复跑方案：m=32768 comm_sd 上翘定性（4×H800）

- **待决问题**：`ANALYSIS_MSWEEP_4xH800.md` §1 末尾的 ⚠ —— fused comm_sd 在
  M=16384→32768 从 1.09 回升到 1.17，且**只有 fused 出现肥尾**。
- **代码**：`comm_comp/rerun_m32768.sh`（驱动）+ `comm_comp/analyze_rerun.py`（判读），
  依赖 `exp3_gemm_rs_fused` 的 `--dump-iters` / `--order` / p50 落盘补丁。
- **跑法**：`cd comm_comp && CUDA_VISIBLE_DEVICES=0,1,2,3 ./rerun_m32768.sh`
  （锁频与空闲预检内置，不通过则 abort；`QUICK=1` 只跑 step 1 冒烟）
- **成本**：约 3 分钟 GPU 时间（7 个点 × ~10 s + 编译）。

---

## 0. 为什么"多跑几次"答不了这个问题

`exp3_gemm_rs_fused` 的 CSV 里 `base_us / ctrl_us / fused_us` 写的是 **mean**
（`exp3_gemm_rs_fused.cu` 的 report 段），而 `make_stats()` 算出的 `p50` / `min`
**被丢弃了**。于是"整个分布右移"和"50 次里有 3 次慢"在数据上**完全同形**：
两者都表现为 mean 抬升 + p95 肥尾。加 `--iters` 只会把同一个被污染的均值估得更准。

所以补丁是前提，不是锦上添花。三处改动：

| 改动 | 解决什么 |
|---|---|
| `p50` / `min` 落 CSV + stdout 第二张表 | 分布右移 vs 尾部 |
| `--dump-iters PRE` 导出逐迭代原始序列（每 rank 一个文件） | 孤立离群点 vs 单调漂移 |
| `--order` 控制 base/ctrl/fused 的计时顺序 | struct_sd<1 是真的还是跑序偏置 |

CSV 新列**追加在原有 16 列之后**，`plot_msweep.py` 用的是 `DictReader`，旧图表脚本不受影响。

新增派生列：

- `comm_sd_p50 = fused_p50 / ctrl_p50` —— 免疫尾部的同一个税率。
- `comm_sd_skew = fused_mean / max_r(ctrl_mean)` —— fused 本来就不可能早于**最慢**
  peer 产出完成，用自己的 ctrl 当分母等于让每个 rank 替 peer 的离散买单。
  20260811 数据按这个口径重算：m=16384 是 1.074、m=32768 是 1.146
  （原 1.089 / 1.167）——**上翘并没有被 skew 归一化吃掉**，所以静态 skew 已可排除。
- `comm_abs_us = fused − ctrl` —— **跨 shape 唯一可比的量**（见 step 2）。

---

## 1. step 1：分布右移，还是尾部？

M=16384 与 M=32768 两点，`--iters 300 --warmup 30`（原轮是 50/5，尾部只有 2–3 个样本）。

从 20260811 数据先算出的 `p95 − mean`（µs），说明这一步问对了地方：

| | base | ctrl | fused |
|---|---|---|---|
| m=16384 | +191 / +136 / +219 / +205 | +34 / +27 / +39 / +37 | **+13 / +4 / +12 / +14** |
| m=32768 | +291 / +249 / +642 / +413 | +89 / +32 / +79 / +74 | **+1089 / +734 / +743 / +1096** |

16384 上 fused 分布紧到几乎无方差，32768 突然 +14%——是相变，不是噪声。

**判据**（`analyze_rerun.py` 自动给）：

| Δcomm_sd(p50) | Δcomm_sd(mean) | 结论 |
|---|---|---|
| <0.02 | >0.04 | **TAIL**。大 M 平台成立，§1 改写为"平台延续，32768 起 fused 出现肥尾" |
| ≥0.04 | — | **SHIFT**。真实成本，⚠ 升级为结论，进 step 2 定位 |
| 其他 | — | **MIXED**，两者叠加 |

## 2. step 2：成本跟 tile 数走，还是跟 kernel 时长走？

固定 K 时 tile 数与时长锁死（都 ∝ M·N），**必须动 K 才能拆开**：

| shape | tiles | pull 量 | ~时长 | 作用 |
|---|---|---|---|---|
| 32768×8192×**8192** | 8192 | 384 MiB | 7.9 ms | 异常点 |
| 32768×8192×**4096** | 8192 | 384 MiB | 4.0 ms | 保 tile/保量，**砍时长** |
| 16384×8192×**16384** | 4096 | 192 MiB | 7.4 ms | 保时长，**砍 tile/砍量** |
| 16384×8192×**8192** | 4096 | 192 MiB | 3.7 ms | 基准 |

⚠️ **这四行之间只能比 `comm_abs_us`，不能比 `comm_sd`**：K 变了分母就变了，比值不可比。

| 哪个对照保住了异常 | 结论 |
|---|---|
| 保 tile 那行（K=4096） | **TILE/VOLUME BOUND**：flag 协议 / fetch 路径的规模项，下一步上 Nsight 看 fetch warp 的 system-scope 轮询 |
| 保时长那行（K=16384） | **DURATION BOUND**：迭代内 skew 或时钟/热漂移——**这是 harness 属性，不是 flux 设计属性**，必须如实标注 |
| 两行都保住一半 | 两者叠加，再加 shape 也拆不开，只能上 profiler |

## 3. step 3：struct_sd < 1 是真的吗？

计时顺序原本写死 `base → ctrl → fused`，**先跑的那个吸收所有残余 ramp-up**，
于是 base 被抬高、`struct_sd = ctrl/base` 被压到 1 以下。注意 20260811 数据里
base 在两个 M 上都有 +4~6% 的尾巴而 ctrl 没有——像极了这个效应。

反序（`--order fused,ctrl,base`）复跑同两个点：

| 观察 | 结论 |
|---|---|
| struct_sd 从 ~0.96 回到 ~1.00 | **跑序偏置**。撤回主报告 §1.4 的"大 M 下略快 0.96（stage 少一档）"，改为"免费，在跑序噪声内" |
| struct_sd 反序后仍 <1 | **真实**，stage/carveout 的解释成立 |

## 4. step 4：尾巴长什么样（自动，无需额外跑）

从 step 1/2/3 各点的 `iters_*.rank*.csv` 里读 fused 序列，报每 rank 的
p50 / mean / max / 超 p50 5% 的迭代占比 / 前 1/3 与后 1/3 的中位数。

| 观察 | 结论 |
|---|---|
| 慢迭代零散分布，前后 1/3 中位数差 <3% | 调度/skew 离群，无漂移 |
| 后 1/3 显著慢（>3%） | **DRIFT**：热/功耗，或跨迭代状态累积（flag 没清干净）——是 harness bug，修掉之前 m=32768 的任何数字都不可引用 |

---

## 5. 结果回填

- **step 1 = TAIL** → 改 `ANALYSIS_MSWEEP_4xH800.md` §1 最后一条与主报告 §1.4；
  `total 最小点在 m=16384` 这句需要按 p50 重算（很可能变成"16384 与 32768 打平"）。
- **step 1 = SHIFT** → §1 的 ⚠ 与"待办"删掉，换成 step 2 的定性结论；§3 的可用域表
  右下角"✅ 甜区"要加上界（甜区在 m=32768 结束）。
- **step 3 = 跑序偏置** → 主报告 §1.4 第一条 bullet 与 §三总结里的
  "大 M 下因 stage 少一档略快 0.96" 必须撤。
- 无论结论如何，`comm_sd_skew` 口径应写进主报告方法学：**用自己的 ctrl 当分母会
  高估通信税**，这条对 K-sweep 的数字同样适用。

## 6. 两个坑

- **不要用 `run_all.sh` 复跑**：它会重跑 exp1/exp2 整套（~40 min）且新建 results 目录，
  对这个问题零价值。但**空闲 + 锁频 1830 的预检必须照做**，否则与原轮不可比——
  `rerun_m32768.sh` 已内置同一套预检。
- **每点跑完必须 `pkill exp3_gemm_rs_fused`**：它 fork 多进程，主进程退出后残余 rank
  会继续占卡污染下一个点。驱动脚本的 `run_point` 已内置。
