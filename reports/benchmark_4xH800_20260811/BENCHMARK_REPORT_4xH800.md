# comm_comp + signalling 4×H800 基准测试报告

- **日期**：2026-08-11
- **硬件**：4× NVIDIA H800 (80 GiB HBM3)，全互联 NVLink (NV8，单机 NVSwitch)
- **软件**：CUDA 12.9 / nvcc V12.9.41，驱动 535.161.08，CUTLASS 4.6 (submodule)
- **编译**：`-arch=sm_90a`（WGMMA/TMA 需 `a` 后缀）
- **代码版本**：git `0cdb385` — *comm_comp: extend M-sweep to 16384/32768*（含 `06cb8b0` 方法学硬化）
- **锁频**：GPU 0–3 SM 锁定 **1830 MHz**（env.txt / clocks_after.txt：跑前跑后均 1830，与上一轮 `06cb8b0` 干净跑一致）

## 方法学

- **串行执行**：comm_comp 与 signalling 都在 GPU 0–3 上**先后**跑，绝不重叠；每套跑前确认 8 卡 0% 利用、0 进程。
- **`06cb8b0` 硬化**：`run_all.sh` 预检目标卡空闲+时钟锁定（不通过 `abort`）；exp1 跑完复测 alone 基线并报 drift（>5% 标记不可信）；exp3 fused 加 RS-disabled control，把 `struct_sd=ctrl/base` 与 `comm_sd=fused/ctrl` 分离归因；signalling 在 pull_kernel 的 t0 后加 grid barrier 修正 pull 系统性欠计时。
- **`0cdb385` 扩展**：exp3 v1（epilogue）与 v2（fused）新增 **M-sweep**（M=64…32768，K=8192），验证大 M 下 wave 饱和后 comm_sd 是否走平。
- **【2026-08-12 追加，来自 m=32768 复跑】三条方法学教训，适用于本报告全部 fused 数字**：
  1. **`iter_us` 是均值，且 CSV 当时不记 p50** —— 尾部会直接漏进每一个比值。50 iters
     的窗口里 2–3 次离群就足以造出一个看起来结构化的效应（m=32768 的 1.17 即如此）。
     现已补 p50/min 落盘与 `--dump-iters` 逐迭代序列。
  2. **`comm_sd = fused/ctrl` 用本 rank 的 ctrl 当分母会高估通信税** —— fused 不可能
     早于**最慢** peer 完成，用 `max_r(ctrl_r)` 归一化后 m=16384/32768 由
     1.089/1.167 降为 1.074/1.146。**这条对 K-sweep 的数字同样适用。**
  3. **跑序有偏**：三个变体固定 `base→ctrl→fused`，先跑者吸收残余开销，
     使 `struct_sd = ctrl/base` 系统性偏低约 0.008。现可用 `--order` 对照。
  4. **长套件末端的点不可与短跑对比**：fused 是功耗最高的变体，40 分钟套件跑到
     最后一个点时的热态与冷机短跑不可比（这是 m=32768 假象最可能的成因，未证实）。
- 所有实验 `rc=0`（无超时/挂死），`--verify` 全通过（`signal` 模式无 payload，记 `skip`，属设计）。exp1 基线 drift <0.4%（远低于 5%），S_c 可信。
- **复现性**：与上一轮干净跑（`06cb8b0`）对比，8192³ GEMM 648 vs 645 TFLOP/s、fused total_sd 1.07 vs 1.07 / 3.06 vs 3.06 / 8.7 vs 8.7——核心数字一致。

原始产物：
- comm_comp：`comm_comp/results_20260811_065005/`（9 个 CSV + 日志 + env + manifest）
- signalling：`signalling/signalling_4xH800.csv`（150 行，5 mode × 6 fanin × 6 size）

---

## 一、comm_comp：通信/计算干扰实验

GEMM = CUTLASS 3.x sm90 fp16（fp32 累加，tile 128×256×64，cluster 2×1×1），单进程多卡、rank r == device r，P2P 走 `cudaDeviceEnablePeerAccess` + UVA。

### 1.1 实验一：CE 搬运对独立 GEMM 的影响（`exp1_ce_interference`）

`S_c = t_overlap / t_alone`（>1 = CE 流量拖慢 GEMM）。基线 drift <0.4%（可信）。

**默认扫描（msg=64 MiB）关键行**

| GEMM (M=N=K) | pattern | S_c | alone TFLOP/s | ovl TFLOP/s | CE ovl GB/s |
|---|---|---|---|---|---|
| 8192 | pull | 1.014 | 648.0 | 639.0 | 173 |
| 8192 | push | 1.013 | 648.0 | 639.6 | 173 |
| 8192 | allgather | 1.021 | 648.0 | 634.5 | 373 |
| 8192 | bystander | 1.007 | 648.0 | 643.7 | 506 |
| 8192 | engine-only | 1.003 | 648.0 | 645.8 | 164 |
| 8192 | local | 1.062 | 648.0 | 610.3 | 107 |
| 4096 | pull / push / allgather / bystander / engine-only | 1.00–1.06 | 645.7 | 609–648 | 167–498 |
| 2048 | local | 2.23 | 573.5 | 257.4 | 977 |

**消息粒度扫描（GEMM=8192³）**：msg≥4 MiB 后 S_c 稳定 1.01–1.03、与 msg 无关；1 MiB 小消息 CE 喂不饱，被标 `[!]`（S_c 为下界）。

**读法**
- **8192³（计算 roofline 648 TFLOP/s）下，跨卡 CE 通信对 GEMM 几乎无影响**：pull/push/allgather/bystander/engine-only 的 S_c 全在 **1.00–1.06**。`bystander`=1.007、`engine-only`=1.003——对照行为正确。计算 bound 的大 GEMM 与 CE 引擎在不同资源上，互不争抢。
- **唯一代价来自本卡 D2D 争本地 HBM**（`local`）：8192³ S_c=1.06，CE 带宽从 ~173 塌到 107 GB/s；2048³ 时 S_c 高达 2.23（小 GEMM 窗内 D2D 相对占比大）。

### 1.2 实验二：搬运粒度 vs 计算效率（`exp2_ag_gemm_granularity`）

真实数据依赖的 AG+GEMM 流水线，扫描每远端 shard 的 chunk 数 S。world=4。

**pull / 3 comm streams，8192³**

| S | t_full µs | t_seg µs | seg_eff | t_comm µs | comm GB/s | t_ovl µs | e2e TFLOP/s |
|---|---|---|---|---|---|---|---|
| 1 | 1784 | 1367 | 1.305 | 579 | 174.0 | 1805 | 609.2 |
| 2 | 1784 | 1511 | 1.181 | 579 | 174.0 | 1707 | **644.3** |
| 4 | 1784 | 1635 | 1.091 | 582 | 172.8 | 1827 | 601.9 |
| 8 | 1784 | 2680 | 0.666 | 582 | 173.1 | 2822 | 389.6 |
| 16 | 1784 | 5218 | 0.342 | 585 | 172.2 | 5289 | 207.9 |
| 32 | 1784 | 10350 | 0.172 | 613 | 164.2 | 10497 | 104.7 |

16384×8192×8192 同趋势，S=2 最优 e2e 665.5 TFLOP/s。serial(1 stream)/push(3 stream) 与 pull 在 e2e 上几乎重合（S=2 时 640–644）。

**读法**：8192³ roofline 648 TFLOP/s，AG 后最优 e2e **644 TFLOP/s（S=2）**，通信几乎全掩藏。代价全在分段 tile 量化损失（`t_seg`）：S≥8 后爆炸，S=32 掉到 105 TFLOP/s。**最优 S=2**，flux `SPLIT==1` 合理。comm 并行度与方向（pull/push、1/3 stream）都不是瓶颈。

### 1.3 实验三 v1：epilogue 远端写（`exp3_epilogue_remote`）

`slowdown` 相对同 epilogue 的 `local` 基线。

**K-sweep（M=N=8192）**

| K | epi | local µs | remote sd | scatter-local sd | scatter sd |
|---|---|---|---|---|---|
| 8192 | tma | 1705 | 1.24 | 0.95 | 1.16 |
| 8192 | nosmem | 1965 | 2.97 | 0.95 | 2.54 |
| 2048 | tma | 457 | 2.07 | 0.98 | 1.90 |
| 2048 | nosmem | 613 | 9.32 | 1.04 | 7.35 |
| 512 | tma | 138 | 6.53 | 1.09 | 5.35 |
| 512 | nosmem | 335 | 16.98 | 1.13 | 13.09 |

**M-sweep（K=8192，tma，本轮新增）**

| M | local TFLOP/s | remote sd | scatter sd |
|---|---|---|---|
| 64 | 104 | 1.06 | 3.92 |
| 128 | 208 | 1.14 | 3.99 |
| 256 | 392 | 1.27 | 3.89 |
| 512 | 604 | 1.45 | 3.19 |
| 1024 | 654 | 1.38 | 1.98 |
| 2048 | 676 | 1.27 | 1.49 |
| 4096 | 652 | 1.27 | 1.29 |
| 16384 | 645 | 1.24 | 1.21 |
| 32768 | 662 | 1.23 | 1.22 |

**读法**
- **远端写代价随算术强度下降急剧放大**（K-sweep）：tma remote 1.24→2.07→6.53；nosmem 2.97→9.32→16.98。这是 flux 在 sm90 刻意不做 epilogue 远端写的原因。
- **M-sweep 揭示 scatter 模式的小 M 灾难**：`scatter`（GEMM+RS 写模式：1/world 本地 + 其余远端）在小 M 下 slowdown 高达 **3.9×（M=64–256）**——远端写分段在微小 GEMM 里占比压倒性；M≥4096 后收敛到 ~1.2。`remote`（D 整体在远端）的 sd 则在 M=512 达峰 1.45 后回落到 ~1.23。
- local roofline 在 M≥1024 走平 ~650–676 TFLOP/s。
- `scatter-local`（分段但全本地）**仅在 M≥2048 时 sd ≤1.1**；小 M 下它和 scatter 一样爆炸（M=64–256：3.7–3.9×，与 scatter 几乎相等）——即小 M 的 scatter 灾难**全部来自分段本身**（4 个子 GEMM 各撞 ~80 µs 的 kernel 地板），远端性一分不收费；大 M 下归因才反转为「分段免费、代价全在远端性」。nosmem remote 全程 2.7–3.5×（非 128-bit 向量化的远端写更贵）。

### 1.4 实验三 v2：flux sm90 融合 GEMM+RS（`exp3_gemm_rs_fused`）

融合 pull 式 RS（flux sm90 真实设计）。RS-disabled control 分离代价：`struct_sd=ctrl/base`（结构改造）、`comm_sd=fused/ctrl`（通信）、`total_sd=fused/base`。world=4，多进程（fork+cudaIPC）。

**K-sweep（M=N=8192）**

| K | struct_sd | comm_sd | total_sd | pull GB/s |
|---|---|---|---|---|
| 8192 | 0.96 | 1.11 | 1.07 | 52.8 |
| 2048 | 1.02 | 2.99 | 3.06 | 72.3 |
| 512 | 1.03 | 8.45 | 8.73 | 87.5 |

**M-sweep（K=8192，本轮新增，4 rank 均值）**

| M | struct_sd | comm_sd | total_sd | pull GB/s |
|---|---|---|---|---|
| 512 | 1.00 | 2.19 | 2.18 | 26.5 |
| 1024 | 1.04 | 1.73 | 1.81 | 33.8 |
| 2048 | 1.08 | 1.23 | 1.33 | 47.5 |
| 4096 | 0.98 | 1.15 | 1.13 | 51.0 |
| 8192 | 0.96 | 1.11 | 1.07 | 52.8 |
| 16384 | 0.96 | 1.09 | **1.05** | 54.1 |
| 32768 | 0.97 | ~~1.17~~ | ~~1.13~~ | 51.0 |

> **⚠ 2026-08-12 修订**：m=32768 的 1.17 / 1.13 是本轮的测量假象（50 iters /
> 5 warmup）。复跑（300 iters / 30 warmup，同机同频）得 comm **1.098**、
> total **1.071**，同口径的 m=16384 为 1.105 / 1.079——**两点打平，平台延续，
> 无上翘**。struct_sd 的 0.96 亦未复现（正序 0.975–0.980）。
> 详见 `../benchmark_4xH800_20260812/ANALYSIS_RERUN_M32768.md`。

**读法**
- **结构改造全程免费**：struct_sd 在所有 M、K 上都是 0.96–1.08——把 2 个空转 warp（CLC 调度/MainloopAux）改成 RS Fetch/Reduce、加 swizzle、smem carveout，本身不亏不赚。~~（大 M 下因 stage 少一档略快 0.96）~~ **【2026-08-12 修订】括号内的"倒赚 4%"撤回**：复跑正序 0.975–0.980、反序 0.980–0.991，0.96 未复现；且 struct_sd 本身带跑序偏置（本实验固定 base 先跑，先跑者吸收残余开销，反序后平均 +0.008）。结论收敛为"免费，在跑序噪声内"。
- **全部代价来自通信**（comm_sd）。M-sweep 给出清晰的「随 M 增长通信被掩藏」曲线：
  - 小 M（512）comm_sd=2.19——GEMM 太短，固定 flag 同步 + NVLink 拉取 + 等最慢 peer 尾部无处藏；
  - M↑ comm_sd 单调下降，**M=8192→16384 走平（1.11→1.09）——这正是 `0cdb385` 要验证的「wave 饱和后大 M 平台」**；
  - ~~**M=32768 comm_sd 回升到 1.17**——平台并非无限延续，尾部方差开始抬头。total_sd 最小点在 M=16384（1.05）。~~ **【2026-08-12 已推翻】** 复跑得 comm 1.098（mean）/ 1.099（p50），超 p50 5% 的迭代占比 0%，肥尾消失；同口径 M=16384 为 1.105，**两点打平，平台延续到 32768**。绝对残差 C=fused−ctrl 从本轮的 74→137 ns/tile（超线性，当初的报警依据）变为复跑的 85→82 ns/tile（线性），与 ctrl 同比例增长，所以走平是结构性的。成因未定：短 warmup 已排除，剩下"长套件末端热累积"（本轮 m=32768 是 40 分钟套件的最后一点），未证实。
- `pull_gbps` 26.5→54 随 M 上升、M≥4096 后饱和 ~51–54——带宽在 M-sweep 里不是瓶颈（远未打满 NVLink），瓶颈是同步/驻留 + 尾部。
- **与 v1 对照**：8192³ 下 epilogue 远端写 sd 1.24（tma）/2.97（nosmem），融合 pull 式 RS total_sd 1.07——flux 在 sm90 选 pull 而非 epilogue 远端写，在 8192³ 上净占优（1.07 < 1.24/2.97），且不破坏 epilogue TMA 路径。低 K 下两者都很贵。

---

## 二、signalling：push vs pull 完成信令开销（`signal_fanin`）

target 卡计时，其余 3 卡 source，逻辑源 round-robin。本轮用修正后的 pull 计时（t0 后 grid barrier）。所有 push/pull/post/push_fence `verify=ok`。

### 2.1 纯 fan-in 下限与协议形状（`signal`/`post`，p50 µs）

| fanin | signal p50 | post p50(4K) | post p50(4M) |
|---|---|---|---|
| 1 | 4.89 | 4.87 | 4.87 |
| 8 | 5.07 | 5.06 | 5.06 |
| 32 | 5.05 | 5.76 | 5.88 |

- `signal`（纯 fan-in）p50 ≈ 4.9–5.1 µs，随 P 几乎不变；`post` ≈ `signal`（drain 成本可忽略）。
- 单机 4 卡测不出 MoK 的 NVL72 71-peer fan-in 尾部；P 1→32 p50 仅 +0.16 µs。形状可读，绝对值不可外推 NVL72。

### 2.2 push 系 vs pull（p50 µs，关键点）

| size | fanin | push | push_fence | pull |
|---|---|---|---|---|
| 4K | 1 | 5.12 | 5.26 | 2.56 |
| 4K | 32 | 6.05 | 5.46 | 3.91 |
| 128K | 1 | 11.93 | 11.94 | 30.97 |
| 128K | 32 | 29.58 | 29.69 | 34.00 |
| 1M | 1 | 60.4 | 60.4 | 234.7 |
| 1M | 32 | 208.5 | 208.3 | 246.5 |
| 4M | 1 | 226.5 | 226.5 | 931.5 |
| 4M | 32 | 861.2 | 861.7 | 977.2 |

**读法（逐条对质 README 预期）**
1. `signal` 随 fanin 缓增、p99 略放大：成立（4 卡上很轻）。
2. `post` − `signal` ≈ 0：成立，drain 免费。
3. `push` 随 size 增长、`signal`/`post` 不随：成立（push 4K→4M 从 5.1 涨到 227 µs）。
4. `push` vs `pull`：与 MoK「push 远贵于 pull」**方向相反**——单机 NVLink 上 **push 在中/大消息上快于 pull**（4M/P=1：push 227 vs pull 932，快 ~4×）；仅 4K 极小消息 pull 略快（2.56 vs 5.12，因 pull 无 go 握手 RTT、单次本地判定完成）。原因：pull 用 `ld.global.cv` 每轮强制过链路读远端，大 size 读带宽成瓶颈；push 是 store 流，NVLink 上效率更高。**MoK 的 push 代价来自 NVL72 跨节点 fan-in + 远端 flag 协议税，非 store 本身**——单机 4 卡测不到，不可外推。
5. `push_fence` − `push` ≈ 0：成立。各点 p50 差 <0.3 µs——sm90 上融合 `st.release.sys` 与手写 `__threadfence_system`+普通 store **无可见差价**。
6. verify 全 ok：可见性语义正确。

---

## 三、总结

### 方法论
- 串行执行 + `06cb8b0` 硬化（预检空闲/锁频、基线 drift 复测、fused struct/comm 分离、pull 计时修正）后数据自洽：drift <0.4%、对照项行为正确、fused slowdown ≥1。本轮复跑核心数字与上轮干净跑一致（648 vs 645 TFLOP/s，fused total_sd 1.07/3.06/8.7 不变），复现性确认。

### comm_comp
- **8192³ 计算 bound GEMM 与 CE 通信互不干扰**：所有跨卡 CE 模式 S_c ≈ 1.00–1.06。唯一代价来自本卡 D2D 争本地 HBM（local S_c 1.06，CE 带宽塌到 107 GB/s）。
- **AG+GEMM 最优粒度 S=2**（e2e 644 TFLOP/s ≈ 648 roofline）；S≥8 后分段量化损失吃掉一切。comm 并行度/方向非瓶颈。
- **epilogue 远端写在 sm90 代价巨大**（K-sweep：tma 1.24→6.5，nosmem 3.0→17；M-sweep：scatter 在小 M 达 3.9×）。flux 改 pull 式 RS 是对的。
- **融合 GEMM+RS 代价全在通信**：struct_sd ≈ 0.96–1.08（结构改造免费），comm_sd 随 M 单调下降、**M=8192→16384 走平（1.11→1.09）验证大 M 平台**，~~M=32768 回升（1.17，尾部方差）。total_sd 最小 1.05（M=16384）。~~ **【2026-08-12 修订】M=32768 的回升不成立**——复跑 comm 1.099、total 1.071，与同口径的 M=16384（1.105 / 1.079）打平，**平台一路延续到 32768，M 方向上没有上界**。8192³ 下 total 1.07，仍优于 epilogue 远端写（1.24 tma / 2.97 nosmem）。
- **【2026-08-12 新增】大 M 救不了小 K**：M=32768、K=4096 测得 comm **1.81**（K=8192 时 1.099），M=16384、K=16384 降到 1.049。通信量由 M×N 定死、计算量 ∝ M·N·K，**算术强度是唯一旋钮**，加大 M 不是替代品。

### signalling
- 协议形状与 MoK 一致：`post≈signal`、`push` 随 size 增长、`push_fence≈push`（融合 release-store 与手写 fence 无差价）。
- 绝对方向与 MoK 相反：单机 NVLink 上 push 中/大消息快于 pull（~4×）；仅极小消息 pull 略快。MoK push 代价来自 NVL72 跨节点协议税，单机 4 卡测不到。

### 复现命令
```bash
nvidia-smi -lgc 1830 -i 0,1,2,3
nvidia-smi --query-gpu=index,utilization.gpu --format=csv   # 应全 0

cd comm_comp && make -j4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh      # 预检不通过会 abort，需 FORCE=1 覆盖

cd signalling && make ARCH=sm_90a
CUDA_VISIBLE_DEVICES=0,1,2,3 ./signal_fanin \
  --modes signal,post,push,push_fence,pull \
  --fanin 1,2,4,8,16,32 --sizes 4K,8K,32K,128K,1M,4M --csv signalling.csv
```
