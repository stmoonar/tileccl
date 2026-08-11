# signalling 基准测试结果 — 4×H20 (NVLink, sm90)

> 运行方式：`CUDA_VISIBLE_DEVICES=0,1,2,3 ./signal_fanin --csv results/signal_fanin.csv`
> 复现 MoK（Mixture of Kittens）push vs pull 完成信令开销实验。
> 原始产物：`signalling/results/`（`signal_fanin.csv` + `signal_fanin.log`）

## 环境

| 项 | 值 |
|---|---|
| GPU | 4× NVIDIA H20，target=GPU0，source=GPU1/2/3 |
| 互联 | NVLink NV18 |
| 工具链 | CUDA 12.9，`-arch=sm_90a`（独立编译，不依赖 CUTLASS） |
| 锁频 | SM clock 稳定 1830 MHz；clock64 标定 1829 cyc/µs，前后一致 |
| 采样 | 300 rounds + 50 warmup，256 threads/CTA，全 5 mode × sizes 4K–128K × fanin 1–32 |

全部配置 `verify=ok`，无 FAIL。

## 五种 mode 的 p50 关键值（µs）

### signal（纯 fan-in 下限，无 payload）

| fanin | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| p50 | 2.71 | 2.72 | 2.72 | 2.72 | 2.96 | 3.55 |
| p99 | 2.72 | 2.94 | 2.94 | 2.94 | 3.60 | 3.78 |

P≤8 完全平坦（~2.72µs ≈ ping-pong RTT），P16 起尾部抬头，p99 放大更明显（fan-in 是尾部现象）。

### push（完整 push 协议，p50，随 size 增长）

| fanin\size | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| 1 | 3.33 | 3.53 | 3.95 | 4.77 | 6.62 | 9.91 |
| 8 | 3.35 | 3.75 | 4.16 | 4.99 | 6.84 | 10.15 |
| 32 | 3.62 | 3.93 | 4.67 | 6.19 | 9.08 | 14.61 |

signal/post 不随 size 变，push 随 size 单调涨 → push 必然背上传输时间。

### pull（p50，requester 本地完成，无跨卡 flag）

| fanin\size | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| 1 | 2.27 | 3.16 | 4.93 | 8.47 | 15.45 | 29.39 |
| 8 | 3.18 | 4.09 | 5.97 | 9.79 | 17.30 | 31.91 |
| 32 | 3.61 | 4.56 | 6.38 | 10.09 | 17.59 | 32.47 |

小 S（4K/P1）= 2.27µs ≈ 单程读延迟（无 RTT 加数）；随 size 强烈增长（L2-bypass 每轮重读）。

## 关键对比

| 对比 | 结论 |
|---|---|
| `post − signal` | ≈ 0（全表）→ pre-flag 握手已排空 payload，drain 成本不漏进 signal 窗 |
| `push_fence − push` | p50 基本相等；差异只在大 S/高 P 的 p99 尾部（P32/128K p99：14.81 vs 15.68，+6%）→ 融合 release-store ≈ 手写 fence |
| `push` vs `pull`（同 S,P） | 小 S/低 P pull 赢（4K/P1：3.33 vs 2.27）；大 S push 赢（128K/P1：9.91 vs 29.39，pull 3× 慢）。存在交叉 |

## 与 MoK 结论对质

MoK 在 NVL72（71 peer）上报 push ~103µs vs pull ~18µs（5.8×）。本机 4 卡 H20 测不出绝对值——3 张物理 source 卡、32 路逻辑 fan-in 上限、逻辑源共享物理链路，**71-way fan-in 尾部无法复现**。但协议形状对得上：signal 随 P 缓增且 p99 放大、post≈signal、push 随 size 涨而 signal 不随、push_fence≈push、verify 全 ok。push/pull 绝对倍数远小于 MoK 的 5.8×（小 S 低 P 仅 1.5×，高 P 收敛），因为缺 71-peer fan-in 聚合，差距主要被链路共享而非协议尾部主导。
