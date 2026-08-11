# comm_comp + signalling 4×H800 基准测试报告（干净复跑）

- **日期**：2026-08-10
- **硬件**：4× NVIDIA H800 (80 GiB HBM3)，全互联 NVLink (NV8，单机 NVSwitch)
- **软件**：CUDA 12.9 / nvcc V12.9.41，驱动 535.161.08，CUTLASS 4.6 (submodule)
- **编译**：`-arch=sm_90a`（WGMMA/TMA 需 `a` 后缀）
- **代码版本**：git `06cb8b0` — *Harden benchmark methodology after contaminated-run review*
- **锁频**：`nvidia-smi -lgc 1980 -i 0,1,2,3`，SM 锁定 1980 MHz

## 本次复跑的方法学修正

上一轮结果作废——comm_comp（GPU 0–3）与 signalling（GPU 4–7）**并发**跑，共享 NVSwitch fabric / host，互相污染：8192³ GEMM alone 被压到 400 TFLOP/s（真实 ~645），exp3 fused 甚至出现 slowdown 0.87（"融合比基线快"，物理上不可能）。本轮修正：

1. **串行执行**：comm_comp 与 signalling 都在 GPU 0–3 上**先后**跑，绝不重叠；每套跑前确认 8 卡 0% 利用、0 进程。
2. **代码升级到 `06cb8b0`**，该提交专门硬化方法学：
   - `run_all.sh` 预检目标卡空闲 + 时钟锁定，否则 `abort`（除非 `FORCE=1`）。
   - **exp1**：所有 pattern 跑完后**复测 alone 基线**并打印 drift；drift >5% 则该 shape 的全部 S_c 标记为不可信。
   - **exp3 fused**：加入 **RS-disabled control**（`comm_enabled=false`，fetch/reduce warp 空转、不发 flag），于是 `struct_sd = ctrl/base`（纯结构改造代价）与 `comm_sd = fused/ctrl`（纯通信代价）**分离归因**，不再与 stock-kernel 基线混在一起。
   - **signalling**：pull_kernel 在 t0 后加第二道 grid barrier，保证时间戳 happens-before 任何 CTA 的首次远端 load（修正此前 pull 系统性欠计时）。

所有实验 `rc=0`（无超时/挂死），`--verify` 全部通过（`signal` 模式无 payload 可校验，记 `skip`，属设计）。exp1 基线 drift 均 <1%（远低于 5% 阈值），S_c 可信。

原始产物：
- comm_comp：`comm_comp/results_20260810_121354/`（日志+CSV+env+manifest，已打包 `results_20260810_121354.tar.gz`）
- signalling：`signalling/signalling_4xH800.csv`（150 行，5 mode × 6 fanin × 5 size）

---

## 一、comm_comp：通信/计算干扰实验

GEMM = CUTLASS 3.x sm90 fp16（fp32 累加，tile 128×256×64，cluster 2×1×1），单进程多卡、rank r == device r，P2P 走 `cudaDeviceEnablePeerAccess` + UVA。

### 1.1 实验一：CE 搬运对独立 GEMM 的影响（`exp1_ce_interference`）

`S_c = t_overlap / t_alone`（>1 = CE 流量拖慢 GEMM）。基线 drift：<0.6%（可信）。

**默认扫描（msg=64 MiB）**

| GEMM (M=N=K) | pattern | S_c | alone TFLOP/s | ovl TFLOP/s | CE ovl GB/s |
|---|---|---|---|---|---|
| 2048 | pull | 1.070 | 573.5 | 536.0 | 164 |
| 2048 | push | 1.034 | 573.5 | 554.5 | 163 |
| 2048 | allgather | 1.329 | 573.5 | 431.6 | 372 |
| 2048 | bystander | 1.001 | 573.5 | 573.0 | 497 |
| 2048 | engine-only | 0.998 | 573.5 | 575.0 | 153 |
| 2048 | local | 2.228 | 573.5 | 257.4 | 977 |
| 4096 | pull | 1.025 | 645.7 | 630.0 | 171 |
| 4096 | push | 1.015 | 645.7 | 636.3 | 173 |
| 4096 | allgather | 1.060 | 645.7 | 609.1 | 419 |
| 4096 | bystander | 0.997 | 645.7 | 647.9 | 498 |
| 4096 | engine-only | 1.001 | 645.7 | 645.1 | 167 |
| 4096 | local | 1.120 | 645.7 | 576.6 | 247 |
| 8192 | pull | 1.015 | 645.0 | 635.6 | 174 |
| 8192 | push | 1.013 | 645.0 | 636.9 | 174 |
| 8192 | allgather | 1.021 | 645.0 | 631.4 | 376 |
| 8192 | bystander | 1.004 | 645.0 | 642.7 | 490 |
| 8192 | engine-only | 1.004 | 645.0 | 642.7 | 165 |
| 8192 | local | 1.060 | 645.0 | 608.7 | 107 |

**消息粒度扫描（GEMM=8192³）**

| pattern | msg | S_c | ovl TFLOP/s | CE ovl GB/s | 备注 |
|---|---|---|---|---|---|
| pull | 1 MiB | 1.275 | 507.6 | 102 | `[!]` CE 中途空闲（S_c 为下界） |
| pull | 4 MiB | 1.015 | 637.6 | 176 | |
| pull | 16 MiB | 1.018 | 636.0 | 175 | |
| pull | 64 MiB | 1.013 | 639.0 | 173 | |
| pull | 256 MiB | 1.016 | 637.0 | 176 | |
| allgather | 1 MiB | 1.096 | 590.7 | 172 | `[!]` CE 中途空闲 |
| allgather | 4 MiB | 1.016 | 636.9 | 349 | `[!]` CE 中途空闲 |
| allgather | 16 MiB | 1.021 | 634.0 | 375 | |
| allgather | 64 MiB | 1.026 | 631.1 | 375 | |
| allgather | 256 MiB | 1.024 | 632.4 | 377 | |

**读法（与上一轮污染结论相反）**
- **8192³（计算 roofline 645 TFLOP/s）下，CE 通信对 GEMM 几乎无影响**：pull/push/allgather/bystander/engine-only 的 S_c 全部落在 **1.00–1.06**。`bystander`=1.004、`engine-only`=1.004——对照行为正确（本卡不参与 / CE 不碰本地 HBM 时，GEMM 不受打扰）。
- **上一轮"干扰主要来自 NVSwitch 链路负载"的结论是污染假象**：干净数据表明，计算 bound 的大 GEMM 与 CE 引擎在不同资源上，互不争抢。NVLink 流量本身不拖慢 SM。
- **唯一显出代价的是 `local`（本卡 D2D，读写均打本地 HBM）**：8192³ 下 S_c=1.06，且 CE 带宽从 ~174 GB/s 塌到 107 GB/s——本地 HBM 带宽被 GEMM 吃满，D2D 被饿死。2048³ 时 S_c 高达 2.23（小 GEMM 30 µs 窗内 D2D 相对占比更大）。
- `allgather`（12 legs，完整 AG 流量）在 8192³ 下 S_c 仅 1.021——进一步印证 CE 流量形态不影响计算 bound GEMM。
- 消息粒度扫描：msg≥4 MiB 后 S_c 稳定 ~1.02、与 msg 无关；1 MiB 小消息 CE 喂不饱（launch 开销 > 传输时间），被正确标记 `[!]`，其 S_c 为下界。

### 1.2 实验二：搬运粒度 vs 计算效率（`exp2_ag_gemm_granularity`）

真实数据依赖的 AG+GEMM 流水线，扫描每远端 shard 的 chunk 数 S（S=1 即 flux 默认粒度）。world=4。

**pull / 3 comm streams**

| shape | S | chunk MiB | t_full µs | t_seg µs | seg_eff | t_comm µs | comm GB/s | t_ovl µs | e2e TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|
| 8192³ | 1 | 32 | 1784 | 1367 | 1.305 | 579 | 174.0 | 1805 | 609.2 |
| 8192³ | 2 | 16 | 1784 | 1511 | 1.181 | 579 | 174.0 | 1707 | **644.3** |
| 8192³ | 4 | 8 | 1784 | 1635 | 1.091 | 582 | 172.8 | 1827 | 601.9 |
| 8192³ | 8 | 4 | 1784 | 2680 | 0.666 | 582 | 173.1 | 2822 | 389.6 |
| 8192³ | 16 | 2 | 1784 | 5218 | 0.342 | 585 | 172.2 | 5289 | 207.9 |
| 8192³ | 32 | 1 | 1784 | 10350 | 0.172 | 613 | 164.2 | 10497 | 104.7 |
| 16384×8192×8192 | 1 | 64 | 3606 | 2904 | 1.242 | 1144 | 175.9 | 3724 | 590.4 |
| 16384×8192×8192 | 2 | 32 | 3606 | 2807 | 1.285 | 1145 | 175.8 | 3304 | **665.5** |
| 16384×8192×8192 | 4 | 16 | 3606 | 3068 | 1.176 | 1146 | 175.7 | 3437 | 639.8 |
| 16384×8192×8192 | 8 | 8 | 3606 | 3271 | 1.103 | 1158 | 173.8 | 3664 | 600.2 |
| 16384×8192×8192 | 16 | 4 | 3606 | 5412 | 0.666 | 1164 | 173.0 | 5680 | 387.1 |
| 16384×8192×8192 | 32 | 2 | 3606 | 10470 | 0.344 | 1166 | 172.7 | 10524 | 209.0 |

**对照：serial（1 stream）/ push（3 streams），8192³，e2e TFLOP/s**

| S | pull-3str | serial-1str | push-3str |
|---|---|---|---|
| 1 | 609.2 | 669.0 | 609.8 |
| 2 | 644.3 | 640.5 | 642.0 |
| 4 | 601.9 | 603.1 | 601.8 |
| 8 | 389.6 | 390.4 | 389.8 |
| 16 | 207.9 | 208.3 | 207.6 |
| 32 | 104.7 | 105.0 | 104.8 |

**读法**
- 干净数据下 8192³ GEMM roofline **645 TFLOP/s**（t_full 1784 µs），AG 后最优 e2e **644 TFLOP/s（S=2）**——通信被几乎完全掩藏，e2e ≈ roofline。
- `t_comm`（纯通信）~579 µs、稳定 ~174 GB/s，与 S 几乎无关——单次 AG 只搬 96 MiB，CE 远没跑满。
- **代价全在 `t_seg`（分段 GEMM 的 tile 量化损失）**：S=1 时分段反比不分段快（seg_eff 1.30，本地 shard 先算、掩藏 launch）；S≥8 后 t_seg 爆炸（S=32 时 10.4 ms，是 t_full 的 5.8×），e2e 掉到 105 TFLOP/s。
- **最优点 S=2**（8192³: 644；16384×8192×8192: 666）——比 flux 默认 S=1 略好，再细分收益为负。flux 的 `SPLIT==1`（整 shard 拷贝）在 Hopper 上合理。
- serial / push / pull 三者在 e2e 上几乎重合（S=2 时 640–644）——**comm 并行度（1 vs 3 stream）和方向（pull vs push）都不是瓶颈，分段粒度才是**。

### 1.3 实验三 v1：epilogue 远端写（`exp3_epilogue_remote`）

GEMM epilogue 直接把 D 写远端（flux sm80 风格）vs 写本地（flux sm90 实际采用）。`slowdown` 相对同 epilogue 的 `local` 基线。

| shape (M×N×K) | epi | local µs | remote µs | slowdown | scatter-local sd | scatter sd |
|---|---|---|---|---|---|---|
| 8192³ | tma | 1705 | 2106 | **1.24** | 0.95 | 1.16 |
| 8192³ | nosmem | 1965 | 5845 | **2.97** | 0.95 | 2.54 |
| 8192×8192×2048 | tma | 457 | 947 | **2.07** | 0.98 | 1.90 |
| 8192×8192×2048 | nosmem | 613 | 5714 | **9.32** | 1.04 | 7.35 |
| 8192×8192×512 | tma | 138 | 902 | **6.53** | 1.09 | 5.35 |
| 8192×8192×512 | nosmem | 335 | 5694 | **16.98** | 1.13 | 13.09 |

**读法**
- **远端写在 Hopper 上代价巨大且随算术强度下降急剧放大**：tma 路径 slowdown 1.24（8192³）→ 2.07（K=2048）→ **6.53**（K=512）；nosmem 路径 2.97 → 9.32 → **16.98**。
- 算术强度越低（K 越小），epilogue 占比越大，远端写代价越显——这正是 flux 在 sm90 上**刻意不做 epilogue 远端写**的原因。
- `scatter-local`（按 world 分段但全写本地）slowdown 普遍 ≤1.1，分段本身几乎免费（甚至因更小 working set 略快）；`scatter`（GEMM+RS 写模式）slowdown 与 `remote` 接近，确认代价来自远端写。
- tma 远端写正常工作（未塌速/报错）；nosmem（寄存器直接 st.global，非 128-bit 向量化）远端写代价远高于 tma。
- 干净数据下 slowdown 略低于污染轮（8192³ tma 1.24 vs 1.49），但**结论方向与强度一致**：远端写在低 K 下是数量级的灾难。

### 1.4 实验三 v2：flux sm90 融合 GEMM+RS（`exp3_gemm_rs_fused`）

flux sm90 **真实采用**的设计：epilogue TMA-store 到本地 + 逐 tile 置 system-scope flag，CTA 空闲 warp 变 RS Fetch/Reduce，通信完全在 kernel 内部。本轮新增 **RS-disabled control**，把 slowdown 拆成结构 vs 通信两部分：
- `struct_sd = ctrl/base`：纯 kernel 结构改造代价（静态调度器、swizzle、smem carveout；通信关闭）
- `comm_sd = fused/ctrl`：纯通信代价（flag 发布、NVLink 拉取、fp16 归约、等最慢 peer 尾部）
- `total_sd = fused/base = struct_sd × comm_sd`

world=4，多进程（fork + cudaIPC）。mainloop stages：baseline 4，ctrl/fused 3（DMA smem carveout）；epilogue StagesC=4 StagesD=2（builder 默认，flux 强制 StagesD=1）。

| shape (M×N×K) | rank 均值 | base µs | ctrl µs | fused µs | struct_sd | comm_sd | total_sd | pull GB/s | verify |
|---|---|---|---|---|---|---|---|---|---|
| 8192³ | 0–3 | 1782 | 1706 | 1908 | 0.958 | 1.119 | **1.071** | 52.8 | ok (rel≤0.018) |
| 8192×8192×2048 | 0–3 | 455 | 466 | 1393 | 1.024 | 2.992 | **3.064** | 72.3 | ok (rel≤0.012) |
| 8192×8192×512 | 0–3 | 132 | 136 | 1151 | 1.034 | 8.45 | **8.73** | 87.5 | ok (rel≤0.005) |

**读法（修正上一轮"融合净赚"的假象）**
- 上一轮 8192³ 出现 slowdown 0.87（融合比基线快）是**污染假象**——基线 GEMM 被并发负载拖慢所致。干净数据下 **8192³ total_sd = 1.07**（融合有 ~7% 代价），符合物理预期。
- **结构改造几乎免费**：struct_sd 在三个 shape 上都是 0.96–1.03——把 2 个空转 warp（CLC 调度 warp / MainloopAux warp）改成 RS Fetch/Reduce、加 swizzle、smem carveout，本身不亏不赚（8192³ 甚至因 stage 数少一档略快 0.96）。
- **全部代价来自通信**：comm_sd 1.12（8192³）→ 3.0（K=2048）→ 8.5（K=512）。算术强度下降 → GEMM kernel 变短 → 固定的 flag 同步 + NVLink 拉取 + 等最慢 peer 尾部无法掩藏，直接膨胀 kernel 时间。
- `pull_gbps` 52–88 GB/s：4 卡对称执行时每张卡一边算一边被其它三卡拉，单卡拉取带宽低（远未打满 NVLink），**瓶颈是同步/驻留 + 尾部等待，不是带宽**。
- 与 1.3 对照：8192³ 下 epilogue 远端写 slowdown 1.24（tma）/2.97（nosmem），融合 pull 式 RS total_sd 1.07——**flux 在 sm90 上选 pull 而非 epilogue 远端写，在 8192³ 上确实更省**（1.07 < 1.24/2.97），且不破坏 epilogue TMA 路径。低 K 下两者都很贵。

---

## 二、signalling：push vs pull 完成信令开销（`signal_fanin`）

复现 MoK 的核心观察。target 卡计时，其余 3 卡为 source，逻辑源 round-robin 摊开。本轮已修正 pull 的欠计时（t0 后加 grid barrier）。所有 push/pull/post/push_fence `verify=ok`。

### 2.1 纯 fan-in 下限与协议形状（`signal` / `post`，p50 µs）

| fanin | signal p50 | post p50(4K) | post p50(4M) |
|---|---|---|---|
| 1 | 4.89 | 4.86 | 4.86 |
| 8 | 5.07 | 5.06 | 5.06 |
| 16 | 5.09 | 4.91 | 5.19 |
| 32 | 5.05 | 4.93 | 5.08 |

- `signal`（纯 fan-in，无 payload）p50 ≈ 4.9–5.1 µs，随 P 几乎不变。这是 push 协议下限，含一跳 target→source go 握手单向延迟。
- `post` ≈ `signal`：payload 提前送达确认后再发 flag，差值 ≤0.3 µs → **drain 成本可忽略**，pre-flag 握手工作正常。
- 单机 4 卡测不出 MoK 的 NVL72 71-peer fan-in 尾部；P 1→32 p50 仅 +0.16 µs，p99 5.2→6.0。结论的「形状」可读，绝对值不可外推 NVL72。

### 2.2 push 系 vs pull（p50 µs，关键点）

| size | fanin | push | push_fence | pull |
|---|---|---|---|---|
| 4K | 1 | 5.25 | 5.25 | 1.62 |
| 4K | 32 | 5.23 | 5.25 | 2.67 |
| 128K | 1 | 11.92 | 11.92 | 30.02 |
| 128K | 32 | 29.58 | 29.61 | 32.07 |
| 1M | 1 | 60.4 | 60.4 | 233.7 |
| 1M | 32 | 208.4 | 208.4 | 246.3 |
| 4M | 1 | 226.7 | 226.7 | 930.7 |
| 4M | 32 | 861.5 | 861.2 | 974.9 |

**读法（逐条对质 README 预期）**
1. `signal` 随 fanin 缓增、p99 略放大：成立（4 卡上很轻）。
2. `post` − `signal` ≈ 0：成立，drain 免费。
3. `push` 随 size 增长、`signal`/`post` 不随：成立。push 4K→4M 从 5.2 涨到 227 µs（P=1），完全被 payload 传输时间主导。
4. `push` vs `pull`：与 MoK「push 远贵于 pull」**方向相反**——单机 NVLink 上 **push 普遍快于 pull**，size 越大优势越大（4M/P=1：push 227 vs pull 931，快 ~4×）。原因：pull 用 `ld.global.cv`（`__ldcv`）每轮强制过链路读远端，大 size 时读带宽成瓶颈；push 是 store 流，NVLink 上效率更高且可聚合。**MoK 的 push 代价来自 NVL72 跨节点 fan-in + 远端 flag 协议税，而非 store 本身**——本机 4 卡测不到。这正是 README 强调的「测协议形状，非 NVL72 绝对值」。
5. `push_fence` − `push` ≈ 0：成立。各 (size,fanin) 点 p50 差 <0.3 µs——sm90 上融合 `st.release.sys` 与手写 `__threadfence_system`+普通 store **无可见差价**。
6. verify 全 ok：可见性语义正确。

---

## 三、总结

### 方法论
- 上一轮的污染来自 **comm_comp 与 signalling 并发跑共享 NVSwitch**，把 GEMM alone 压低 35%、并制造出"融合比基线快"的假象。本轮串行 + 升级到 `06cb8b0`（预检空闲/锁频、基线 drift 复测、fused struct/comm 分离归因、pull 计时修正）后，数据自洽：基线 drift <1%、对照项（bystander/engine-only/scatter-local）行为正确、fused slowdown ≥1。

### comm_comp（干净结论）
- **8192³ 计算 bound GEMM 与 CE 通信互不干扰**：所有跨卡 CE 模式 S_c ≈ 1.00–1.06，bystander/engine-only ≈ 1.00。唯一代价来自本卡 D2D 争本地 HBM（local S_c 1.06，CE 带宽塌到 107 GB/s）。上一轮"NVSwitch 链路负载干扰 SM"的结论是假象。
- **AG+GEMM 最优粒度 S=2**（e2e 644 TFLOP/s ≈ 645 roofline，通信几乎全掩藏）；S≥8 后分段量化损失吃掉一切。flux `SPLIT==1` 合理。
- **epilogue 远端写在 sm90 代价巨大**（slowdown 1.24–17×，随算术强度下降放大），flux 改 pull 式 RS 是对的。
- **融合 GEMM+RS 的代价全部来自通信**：struct_sd ≈ 1.0（结构改造免费），comm_sd 1.12/3.0/8.5（随 K 下降暴涨）。8192³ total_sd 1.07，仍优于 epilogue 远端写（1.24 tma / 2.97 nosmem），flux 的 pull 选择在 8192³ 上净占优。

### signalling
- 协议形状与 MoK 一致：`post≈signal`（drain 免费）、`push` 随 size 增长、`push_fence≈push`（融合 release-store 与手写 fence 无差价）。
- 绝对方向与 MoK 相反：单机 NVLink 上 push 普遍快于 pull（大消息快 ~4×）。MoK 的 push 代价来自 NVL72 跨节点 fan-in 协议税，单机 4 卡测不到，不能外推 NVL72 绝对值。

### 复现命令
```bash
# 锁频 + 确认空闲
nvidia-smi -lgc 1980 -i 0,1,2,3
nvidia-smi --query-gpu=index,utilization.gpu --format=csv   # 应全 0

# comm_comp（4 卡，串行，约 1–2 分钟）
cd comm_comp && make -j4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh      # 预检不通过会 abort，需 FORCE=1 覆盖

# signalling（同 4 卡，comm_comp 跑完后再跑）
cd signalling && make ARCH=sm_90a
CUDA_VISIBLE_DEVICES=0,1,2,3 ./signal_fanin \
  --modes signal,post,push,push_fence,pull \
  --fanin 1,2,4,8,16,32 --sizes 4K,8K,32K,128K,1M,4M --csv signalling.csv
```
