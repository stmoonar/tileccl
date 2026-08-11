# comm_comp 基准测试结果 — 4×H20 (NVLink, sm90)

> 运行方式：`CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh`（4 卡环境，QUICK=0 全量）
> 原始产物目录：`comm_comp/results_20260810_053932/`（日志、CSV、env.txt、tar.gz）

## 环境

| 项 | 值 |
|---|---|
| GPU | 4× NVIDIA H20，sm_90，78 SM/GPU |
| 互联 | NVLink NV18（全互联，4 卡间两两 18 链路） |
| 工具链 | CUDA 12.9 (nvcc V12.9.41)，`-arch=sm_90a`，CUTLASS submodule `f94ec46` |
| 锁频 | `nvidia-smi -lgc 1980 -i 0,1,2,3`；运行全程 SM clock 稳定 1830 MHz，前后一致，无 throttle（`clocks_event_reasons=0x1` GpuIdle） |
| git commit | `06cb8b0` (bench/p2p-ce-vs-tma) |
| GEMM 配置 | fp16 CUTLASS 3.x，`TmaWarpSpecializedCooperative`，tile 128×256×64，cluster 2×1×1（exp3 v2 cluster 1×2×1） |
| 角色分配 | world=4，GEMM/target=GPU0，peer=GPU1/2/3（exp3 v2 为 1 进程/卡 + cudaIPC） |

环境前置检查通过（4 卡利用率 0%、显存 0、SM clock 1830 > 1000）。所有实验 `rc=0`，`verify=ok`（exp3 v2 相对误差 <2%），无 FAIL。

---

## 实验一：CE 搬运对独立 GEMM 的影响（`exp1_ce_interference`）

GEMM 在 GPU0 跑，同时 CE 在各 rank 间搬运**与 GEMM 无关**的数据。`S_c = t_overlap / t_alone`。

### 默认扫描（msg=64M，shapes 2K/4K/8K）

| shape | alone TFLOP/s | pull S_c | push S_c | allgather S_c | bystander S_c | engine-only S_c | local S_c | 基线漂移 |
|---|---|---|---|---|---|---|---|---|
| 2048³ | 109.8 | 1.014 | 1.005 | **1.087** | 1.000 | 1.001 | 1.019 | −0.3% |
| 4096³ | 132.5 | 1.004 | 1.000 | 1.011 | 1.000 | 1.000 | **1.029** | 0.0% |
| 8192³ | 139.4 | 1.000 | 1.000 | 1.001 | 1.000 | 1.000 | 1.001 | +0.1% |

### 消息粒度扫描（8192³ GEMM，pull / allgather，1M–256M）

| msg | pull S_c | pull ovl GB/s | allgather S_c | allgather ovl GB/s |
|---|---|---|---|---|
| 1M | 1.001 [!] | 140 | 1.001 [!] | 227 |
| 4M | 1.000 [!] | 328 | 1.000 [!] | 537 |
| 16M | 1.000 | 394 | 1.002 | 822 |
| 64M | 1.000 | 392 | 1.002 | 883 |
| 256M | 1.000 | 390 | 1.002 | 866 |

`[!]` = CE 在计时窗中途空闲（宿主喂不动），S_c 为下界（仍 ≈1）。

### 结论

- **CE 引擎对独立 GEMM 基本免费**：pull / push / engine-only / bystander 在 4K/8K 上 S_c 全部 ≈ 1.000。`engine-only ≈ 1.00` 说明 CE 引擎本身不占用计算资源。
- **唯一可见的竞争是本地 HBM 带宽**：`local`（D2D 读写都在本地 HBM）在 4096³ 上 S_c=1.029；`allgather` 在最小的 2048³ GEMM 上 S_c=1.087（小 GEMM 对 HBM 抖动更敏感）。大 GEMM（8192³）即使全模式 AG 流量也只 S_c=1.001。
- 消息大小对 S_c 无影响（1M–256M 均 ≈1）；allgather 的重叠带宽随 msg 增长到 ~880 GB/s 后饱和。
- 基线漂移全部 ≤0.3%，数据可信。

---

## 实验二：搬运粒度 vs 计算效率（`exp2_ag_gemm_granularity`）

真实数据依赖的 AG+GEMM 流水线，扫描每远端 shard 的 chunk 数 S（`S=1` 即 flux 默认粒度）。

### 8192³（shard M=2048，AG 96 MiB），pull 模式

| S | t_full | t_seg | seg_eff | t_comm | GB/s | t_ovl | ovl TFLOP/s | ovl_eff | bub% |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7889 | 8173 | 0.965 | 264.8 | 380 | **8204** | 134.0 | 0.884 | 0.4% |
| 2 | – | 9374 | 0.842 | 266.8 | 377 | 9394 | 117.0 | 0.925 | 0.2% |
| 4 | – | 9406 | 0.839 | 264.9 | 380 | 9446 | 116.4 | 0.849 | 0.4% |
| 8 | – | 9530 | 0.828 | 268.3 | 375 | 9564 | 115.0 | 0.876 | 0.3% |
| 16 | – | 19001 | 0.415 | 319.4 | 315 | 19086 | 57.6 | 0.733 | 0.4% |
| 32 | – | 37996 | 0.208 | 462.8 | 218 | 38131 | 28.8 | 0.708 | 0.4% |

### 模式对比（8192³，S=1 / S=32 的 t_comm GB/s）

| 模式 | S=1 GB/s | S=32 GB/s |
|---|---|---|
| pull（每 peer 一条 comm stream） | 380 | 218 |
| push（peer 侧 CE 推送） | 383 | 342 |
| serial（单串行 comm stream） | 348 | **78** |

### 结论

- **甜点 = S=1（flux 默认的最粗粒度）**：t_ovl 8204µs vs t_full 7889µs，仅 ~4% 气泡，且通信（265µs）几乎完全藏在计算（8173µs）背后。`bub%` 仅 0.4%。
- **更细的 chunk 严格更差**：S=16 时 `seg_eff` 掉到 0.41，S=32 掉到 0.21——分段本身把 GEMM tile 量化打崩，t_ovl 膨胀到 19ms / 38ms。每 chunk 的信号成本（event record + stream wait + launch）+ tile 量化损失压过了"更早启动计算"的收益。
- push ≈ pull（通信带宽 383 vs 380 GB/s，方向几乎无差）。
- 串行 comm stream 在细粒度下 t_comm 崩塌（S=32 仅 78 GB/s），但 **t_ovl 几乎不变**——因为通信始终藏在计算背后，串行化只影响通信自身的完成时间，不暴露到端到端。
- 16384×8192×8192 大形状同构（S=1 仍最优，t_ovl 16387µs vs t_full 15486µs）。

---

## 实验三 v1：epilogue 远端写对计算的影响（`exp3_epilogue_remote`）

同一 GEMM，只换 D 指针与 epilogue 形态。`slowdn = iter / 同-epilogue 的 local 基线`。

| K | epilogue | local | remote | scatter-local | scatter(scatter=1/world 本地+远端) |
|---|---|---|---|---|---|
| 8192 | tma | 1.000 | **1.000** | 1.038 | 1.045 |
| 8192 | nosmem | 1.000 | 1.031 | 1.041 | 1.129 |
| 2048 | tma | 1.000 | **1.008** | 1.046 | 1.077 |
| 2048 | nosmem | 1.000 | **1.478** | 1.052 | 1.561 |
| 512 | tma | 1.000 | **1.036** | 1.071 | 1.185 |
| 512 | nosmem | 1.000 | **4.061** | 1.089 | 3.445 |

### 结论

- **TMA 远端写几乎不打扰计算**：`tma remote` 在所有 K 上 slowdn 仅 1.000–1.036——TMA 描述符建在 peer UVA 指针上工作正常（verify ok），flux "未踩过的 TMA-store-to-peer 路径" 在 H20 上没出问题。`tma scatter`（GEMM+RS 写模式）代价也仅 1.045–1.185。
- **nosmem（sm80 风格 st.global 远端写）在低算术强度下灾难性**：K=512 时 `nosmem remote` slowdn=**4.06×**，`nosmem scatter`=3.45×；K=2048 时 1.48× / 1.56×；K=8192 才回到 1.03。算术强度越低，epilogue 占比越大，非向量化的远端 store 代价越显。
- **远端写代价是 nosmem 路径特有的，不是 TMA/远端写本身的问题**：同一 remote 模式下 tma ≈ 1.0 而 nosmem 高达 4×。这正面回答了 README 的核心问题——flux 在 sm90 上选 TMA-store-本地 + pull，相比 sm80 的 nosmem 远端写，在低 K 下有据可循。

---

## 实验三 v2：flux sm90 融合 GEMM+RS（`exp3_gemm_rs_fused`）

多进程（1 进程/卡 + cudaIPC），三组对照同环境一次测完：`struct = ctrl/base`（kernel 结构改造税），`comm = fused/ctrl`（通信净代价），`total = fused/base`。

| K | base TFLOP/s | struct(worst) | comm(worst) | total(worst) | fused TFLOP/s | pull GB/s | verify |
|---|---|---|---|---|---|---|---|
| 8192 | 139.6 | 1.001 | **1.025** | 1.026 | 135.9 | 12.4 | ok (rel 0.018) |
| 2048 | 136.4 | 1.004 | **1.074** | 1.078 | 126.4 | 46.4 | ok (rel 0.012) |
| 512 | 124.9 | 1.013 | **2.341** | 2.372 | 52.6 | 77.5 | ok (rel 0.006) |

### 结论

- **结构税可忽略**：`struct` 全部 ≤1.013——ctrl 组（同调度器/swizzle/smem carveout，通信关闭）与 stock baseline 几乎同速，证明 flux 的 kernel 改造本身不收税，归因干净。
- **通信净代价随算术强度下降而剧增**：K=8192 comm 仅 +2.5%（融合 RS 几乎免费）；K=2048 +7.4%；K=512 **+134%**（comm=2.34，fused 1294µs vs base 552µs）。低 K 下通信（flag 同步 + NVLink 拉取 + fp16 归约 + 等最慢 peer 尾部）主导 kernel 时间。
- worst-rank 与均值接近（4 卡对称执行，同步尾部不显著）。
- verify 全 ok，rel 误差 <2%。

### 与 v1 对照（flux sm90 pull vs sm80 nosmem-remote，低 K 下的关键对比）

| K | sm80 风格 nosmem-remote (v1) | sm90 风格 fused-pull (v2 total) | pull 优势 |
|---|---|---|---|
| 8192 | 1.031 | 1.026 | 持平 |
| 2048 | 1.478 | 1.078 | **pull 快 1.37×** |
| 512 | 4.061 | 2.346 | **pull 快 1.73×** |

→ flux 在 sm90 上选 pull 式 RS 而非 epilogue 远端写，**在低算术强度下显著占优**（K=512 快 1.73×），高算术强度下两者都接近免费。设计选择有据。

---

## 总览

| 实验 | 核心结论 | H20 上的数字 |
|---|---|---|
| exp1 | CE 引擎对独立 GEMM 基本免费，唯一竞争是本地 HBM | S_c ≈ 1.00（pull/push/engine-only/bystander）；local 最高 1.029，allgather@2048 1.087 |
| exp2 | 最粗粒度 S=1 最优，更细分段因 tile 量化崩塌 | S=1 t_ovl 仅 +4%；S=32 seg_eff 0.21、t_ovl 4.8× |
| exp3 v1 | TMA 远端写近乎免费；nosmem 远端写在低 K 灾难 | tma remote 1.00–1.04；nosmem remote K=512 达 4.06× |
| exp3 v2 | 融合 RS 通信税随 K 下降剧增；结构税可忽略 | K=8192 comm +2.5%；K=512 comm +134%；pull 比 nosmem-remote 快 1.73× |

**一句话**：在 4×H20 NVLink 上，CE 流量与独立 GEMM 互不干扰；AG+GEMM 应取最粗粒度（flux 默认）；flux 的 sm90 pull 式 RS 设计相比 sm80 epilogue 远端写，在低算术强度下有明确收益（K=512 快 1.73×），在高算术强度下两者都接近免费。

---

## 产物清单（`results_20260810_053932/`）

| 文件 | 内容 |
|---|---|
| `manifest.txt` | 运行清单 + 各实验 rc/耗时 |
| `env.txt` | git/uname/nvcc/nvidia-smi/拓扑/时钟快照 |
| `build.log` | make -j4 编译日志 |
| `exp1_default.{log,csv}` / `exp1_msgsweep.{log,csv}` | CE 干扰 |
| `exp2_{verify,pull,serial,push}.{log,csv}` | 粒度扫描 |
| `exp3_epilogue.{log,csv}` | epilogue 远端写 |
| `exp3_fused.csv` + `exp3_fused_{8192,k2048,k512}.log` | 融合 GEMM+RS |
| `clocks_after.txt` | 跑后时钟（确认无掉频） |
| `results_20260810_053932.tar.gz` | 打包 |
