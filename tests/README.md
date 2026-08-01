# P2P microbench

两个独立的可执行文件，`make` 一次全部构建：

| binary | 回答的问题 |
|---|---|
| `p2p_ce_vs_tma` | 通算融合应该走哪条搬运路径、tile 该多大（按 token 粒度扫） |
| `pk_bw_sweep` | 复现 ParallelKittens 论文（arXiv:2511.13940）Fig.2 / Fig.3 / Tab.1：不同传输机制在不同**消息大小**和**SM 数**下打到的带宽 |

---

## pk_bw_sweep：论文 Fig.2 / Fig.3 复现

三种机制，全部是 push（源 GPU 执行）：

| method | 机制 | 对应论文 |
|---|---|---|
| `ce` | 每条消息一次 `cudaMemcpyPeerAsync`（host 发起，DMA 引擎） | Copy Engine |
| `tma` | kernel 内 `cp.async.bulk`：local gmem → smem → peer gmem，mbarrier 流水 | TMA Op |
| `reg` | kernel 内 `uint4` ld/st 直写 peer VA | Register Op |

两组 sweep：

- **Fig.2（`--mode size`）**：固定总量（默认 1 GiB），把它切成大小为 S 的消息，扫 S 从 128 B 到 1 GiB，看每种机制的带宽曲线。论文结论：CE 需要 ≥256 MB 消息才能到 80% 利用率，TMA 2 KB 就能接近峰值。
- **Fig.3（`--mode sms`）**：固定消息大小，扫参与搬运的 SM（CTA）数，看多少个 SM 能打满互联带宽。论文结论：TMA 约 15 个 SM 饱和，register op 需要约 76 个。CE 不占 SM，作为参考线单独打一行。

```bash
./pk_bw_sweep                                        # 两组 sweep 全跑，1 GiB
./pk_bw_sweep --mode size --sizes 128,2K,64K,2M,64M,1G
./pk_bw_sweep --mode sms --sms 1,2,4,8,16,32 --msg-reg 16K
./pk_bw_sweep --total 256M --iters 20                # 显存紧张时缩小总量
```

主要参数：`--total`（每次测量的总载荷）、`--sizes` / `--sms`（两个 sweep 的扫描点，支持 K/M/G 后缀）、`--stages`（TMA 流水深度 3/4/6/8）、`--reg-threads`（reg kernel 每 CTA 线程数，默认 1024）、`--max-ce-msgs`（小消息时 CE 每轮最多发几条，防止 host 端 enqueue 时间爆炸）。

输出单元格里的标记：

- `*`：CE 载荷被 `--max-ce-msgs` 截短（带宽按实际搬运字节数算，小消息下这正是要测的 per-call 开销）；
- `^`：消息太大装不下 ≥3 级流水，TMA 退化为单 buffer load→store（本地 load 远快于 P2P store，误差很小）；
- `-`：超过 shared memory 预算，TMA 无法承载该消息大小（论文在这个区间直接把曲线画平，对应 H100/B200 的 227 KB 上限）；
- `!`：目的端校验失败。

注意：论文是 NVLink（H100/B200），本机是 PCIe 上的 sm_120，绝对数值不可比，但曲线形状（各机制的饱和粒度 / 饱和 SM 数）是同一组问题。

---

## p2p_ce_vs_tma：Copy Engine vs TMA

用来回答一个问题：**在 PCIe 互联的两张 sm_120 上，tile 粒度的通算融合应该走哪条搬运路径、tile 该多大。**

## 构建

```bash
cd tests
make                    # 默认 -arch=sm_120a，需要 CUDA >= 12.8
make ARCH=sm_120        # 如果工具链不接受 family-specific arch
```

## 运行

```bash
./p2p_ce_vs_tma                                     # 默认扫 1..4096 tokens
./p2p_ce_vs_tma --tokens 1,4,16,64,256,1024,4096 --hidden 7168 --dtype bf16
./p2p_ce_vs_tma --tokens 256 --tile-tokens 8 --stages 4   # 手工指定 tile 粒度
./p2p_ce_vs_tma --methods tma,tma_store --src 0 --dst 1
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--tokens LIST` | 逗号分隔，一次扫出粒度曲线 |
| `--hidden N` / `--dtype` | 决定每 token 字节数；要求 `hidden * sizeof(dtype)` 是 16 的倍数 |
| `--tile-tokens N` | 每个 TMA tile 的 token 数，`0` 表示按 shared memory 预算自动取最大 |
| `--stages N` | TMA 软件流水深度，可选 3/4/5/6/8，默认 4 |
| `--blocks N` | kernel 路径的 CTA 数，`0` 自动（≈ SM 数） |
| `--methods LIST` | `ce,sm,tma,tma_store` 的子集 |
| `--no-check` | 跳过回读校验 |

## 四种搬运路径

| method | 路径 | 含义 |
|---|---|---|
| `ce` | `cudaMemcpyPeerAsync` | DMA 拷贝引擎，不占 SM，但有固定启动开销 |
| `sm` | kernel 内 `uint4` load/store 写 peer 地址 | SM 直写，DeepEP 风格 |
| `tma` | local gmem →(TMA)→ smem →(TMA)→ peer gmem | 与 `ce` 同为 gmem→gmem，公平对比 |
| `tma_store` | smem →(TMA)→ peer gmem | **融合场景的上界**：tile 刚在 shared 里算完直接推出去，省掉回写本地 global 的一跳 |

`tma_store` 的 payload 是每个 CTA 首个 tile 的重复拷贝，目的地内容**不是**源数据的忠实副本，所以它的 `verify` 一栏固定是 `skip`。它只用来量纯写路径的延迟/带宽。

## 输出解读

每个 token 数打印一组：

```
--- tokens=256   payload=3.500 MiB   tile=14336 B (1 tok)   tiles=256   stages=4   blocks=128   smem=59392 B ---
method           avg(us)      p50(us)     BW(GB/s)   verify
ce                 ...          ...          ...        ok
```

- `avg(us)`：背靠背发 `--iters` 次的设备时间均值。launch 开销被流水掉，反映**持续吞吐**。
- `p50(us)`：每次单独同步的一发一收延迟中位数。反映**融合场景真正关心的那个数**——一个 tile 从发起到落地要多久。
- `BW(GB/s)`：由 `avg(us)` 算出。
- `verify`：把目的地拷回 host 与源比对。

看曲线的方式：`p50` 在小 token 数下趋于各方法的固定开销（CE 的 DMA 启动、kernel 的 launch），交叉点就是"tile 小到什么程度以后 CE 不划算"。`BW` 在大 token 数下趋于 PCIe 上限，谁先到顶谁的粒度效率高。

## 需要注意的前提

1. **本测试在验证一个假设**：`cp.async.bulk.global.shared::cta` 的目的地址可以是 `cudaDeviceEnablePeerAccess` 之后映射进统一虚拟地址空间的 peer 显存。如果 TMA 单元走不通 PCIe P2P 通路，会表现为 illegal address 或 `verify` 失败——这两种结果本身都是有效结论。
2. TMA kernel 结尾用的是 `cp.async.bulk.wait_group 0` 而不是 `.read` 变体，确保计时包含写真正落到对端，而不只是源 shared buffer 可以复用。
3. 一旦某个 kernel 触发 sticky fault，CUDA context 就废了，后续方法都会失败。用 `--methods` 单独重跑。
4. `nvidia-smi topo -m` 确认两卡确实是 PCIe（`PHB`/`PXB`/`SYS`）而不是 NVLink，否则结论不适用。
5. 若 P2P 不可用，`ce` 会退化成经 host 中转（仍能跑，但慢一个量级），`sm`/`tma`/`tma_store` 直接标记为不可用。
