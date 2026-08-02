# P2P microbench

四个独立的可执行文件，`make` 一次全部构建：

| binary | 回答的问题 |
|---|---|
| `p2p_ce_vs_tma` | 通算融合应该走哪条搬运路径、tile 该多大（按 token 粒度扫） |
| `pk_bw_sweep` | 复现 ParallelKittens 论文（arXiv:2511.13940）Fig.2 / Fig.3 / Tab.1：不同传输机制在不同**消息大小**和**SM 数**下打到的带宽 |
| `interference_matrix` | 通信对计算、计算对通信的**双向干扰矩阵**(通信方式 x 计算瓶颈类型),以及满载 SM 下三种搬运方式的 progress/starvation 行为 |
| `sync_cost` | 细粒度 pipeline 的同步原语成本:跨卡 flag 单向延迟、fence+signal 尾部开销、远端 atomic 往返、本地 mbarrier 周期 |

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
method           avg(us)      p50(us)      p99(us)     BW(GB/s)   verify
ce                 ...          ...          ...        ok
```

- `avg(us)`：背靠背发 `--iters` 次的设备时间均值。launch 开销被流水掉，反映**持续吞吐**。
- `p50(us)`：每次单独同步的一发一收延迟中位数。反映**融合场景真正关心的那个数**——一个 tile 从发起到落地要多久。
- `p99(us)`:one-shot 延迟的尾部(64 个样本时即最大值)。融合 pipeline 的 consumer 等的是最慢的那个 tile,所以尾部比中位数更接近真实代价;p99 与 p50 差距大说明该路径受调度抖动影响。
- `BW(GB/s)`：由 `avg(us)` 算出。
- `verify`：把目的地拷回 host 与源比对。

看曲线的方式：`p50` 在小 token 数下趋于各方法的固定开销（CE 的 DMA 启动、kernel 的 launch），交叉点就是"tile 小到什么程度以后 CE 不划算"。`BW` 在大 token 数下趋于 PCIe 上限，谁先到顶谁的粒度效率高。

## 需要注意的前提

1. **本测试在验证一个假设**：`cp.async.bulk.global.shared::cta` 的目的地址可以是 `cudaDeviceEnablePeerAccess` 之后映射进统一虚拟地址空间的 peer 显存。如果 TMA 单元走不通 PCIe P2P 通路，会表现为 illegal address 或 `verify` 失败——这两种结果本身都是有效结论。
2. TMA kernel 结尾用的是 `cp.async.bulk.wait_group 0` 而不是 `.read` 变体，确保计时包含写真正落到对端，而不只是源 shared buffer 可以复用。
3. 一旦某个 kernel 触发 sticky fault，CUDA context 就废了，后续方法都会失败。用 `--methods` 单独重跑。
4. `nvidia-smi topo -m` 确认两卡确实是 PCIe（`PHB`/`PXB`/`SYS`）而不是 NVLink，否则结论不适用。
5. 若 P2P 不可用，`ce` 会退化成经 host 中转（仍能跑，但慢一个量级），`sm`/`tma`/`tma_store` 直接标记为不可用。

---

## interference_matrix:通算干扰矩阵

回答的问题:**每种搬运方式,对每类计算瓶颈,每搬 1 字节偷走多少计算;反过来计算又把通信拖慢多少。** 这是通算融合选型真正需要的数,而不是裸带宽。

### 方法学

计算侧是 5 个 persistent 探针 kernel,每个把一种资源打到饱和,并把每个 CTA 的迭代数持续写回 global memory:

| probe | 饱和的资源 | 吞吐单位 |
|---|---|---|
| `mma` | Tensor Core(wmma 链,纯寄存器) | TFLOP/s |
| `ffma` | issue/ALU(FFMA 依赖链,纯寄存器) | GFLOP/s |
| `hbm` | HBM 带宽(工作集远大于 L2 的流式读写) | GB/s |
| `l2` | L2 带宽(所有 CTA 共享约 L2/2 的只读工作集) | GB/s |
| `smem` | shared memory / LSU(依赖式 smem load) | GB/s |

通信侧复用 `pk_bw_sweep` 的三条路径(`ce` / `tma` / `reg`,全部 push)。每个 (probe, comm) 格子测三遍:comm alone、probe alone、两者并发,并发窗口内用计数器快照测探针速率。输出:

- `S_m` = comm alone BW / overlap BW:计算把通信拖慢的倍数;
- `S_c` = probe alone rate / overlap rate:通信把计算拖慢的倍数;
- 探针在 overlap 窗口内计数增量为 0 会打 `[!]`:说明两者根本没有并发(结果无效)。

CTA 布局:probe 与 comm kernel 同卡时,probe 拿 (SM 数 - `--comm-sms`) 个 CTA;`ce` 行 probe 拿全部 SM——这正是 CE 的卖点,也是为什么 `ce` 行的 `S_c` 是纯 memory 系统竞争,而 `tma`/`reg` 行的 `S_c` 还包含让出的 CTA。

### 两种模式

- `--mode matrix`(默认):上述干扰矩阵。
- `--mode starve`:实验 C。一个满占用(SM x 4 个 CTA)的 ffma kernel 常驻源卡,再让每种搬运方式发一次 payload。预期:`ce` 照常完成;`tma`/`reg` 拿不到 CTA slot,饿死到 squatter 退出为止。这是三者 progress model 的本质差别,带宽曲线看不出来。

```bash
./interference_matrix                                  # 全矩阵
./interference_matrix --probes hbm,mma --comms ce,tma
./interference_matrix --probe-dev dst                  # 接收端干扰(实验 E)
./interference_matrix --mode starve
./interference_matrix --csv results.csv                # 机器可读输出
```

主要参数:`--total`(每次通信迭代的 payload,默认 64M)、`--msg`(消息大小,默认 2M,须整除 total)、`--comm-sms`(tma/reg 的 CTA 数,默认 8)、`--probe-sms`(0 = 自动)、`--window-ms`(每格测量窗口,默认 300)、`--probe-dev src|dst`、`--hbm-buf`(hbm 探针工作集,默认 256M)。

### 结果解读与前提

1. **先锁频**:`nvidia-smi -lgc <min>,<max>`(需要 root/管理员)。通信+计算同时跑功耗上升,DVFS 降频会被误读成资源竞争。程序启动时会打印提醒;消费卡锁不了频的话,至少记录 `nvidia-smi --query-gpu=clocks.sm` 并在报告里注明。
2. **CTA 布局是尽力而为**:没有 green context 的话,probe 和 comm 的 CTA 不保证落在不相交的 SM 上,`S_c` 里可能混入同 SM 的 warp 调度竞争。要硬分区可以在 H20(CUDA >= 12.4)上用 green context 改造。
3. 预期的模式(用来校验数据是否合理):`ce` x `mma`/`ffma`/`smem` 应接近 1.0(CE 不碰 SM,也不碰这些资源);`ce`/`tma`/`reg` x `hbm` 都应明显 > 1(源端 HBM 读竞争不可避免);`reg` 行的 `S_c` 应普遍高于 `tma` 行(通信 warp 抢 issue slot 和 LSU);`l2` 行反映链路流量对 L2 的污染。
4. matrix 模式不做数据校验(pk_bw_sweep 已经验证过同样的搬运 kernel);starve 模式下 `under-squat` 时间对饿死的方法约等于 squatter 存活时间,不是传输本身的时间。

---

## sync_cost:同步原语成本

细粒度 pipeline 里每个 tile 的成本 = payload 搬运 + fence + signal + 对端 polling。小 tile 时后三项可能主导,必须单独量化,否则会把同步开销错记在搬运方式头上。

四组测量:

1. **flag ping-pong**:两卡各一个单线程 kernel,用 `st.release.sys` / `ld.acquire.sys` 打乒乓,RTT/2 即单向通知延迟——任何 "tile ready" 信号的下限。跨卡时钟不同步,所以一切数字都来自单卡计时的往返,从不比较两卡时间戳。
2. **producer 序列**:单 warp 向对端写 S 字节 payload,然后分别测三个变体:纯 stores / +`st.release.sys` flag / +`__threadfence_system`+普通 store。差值就是 fence+signal 的边际成本;它随 S 增长的部分是 fence 排空 outstanding P2P writes 的代价。
3. **远端 atomic 往返**:依赖链式 `atomicAdd_system` 打对端计数器,每 op 即完整往返——MoE 计数器/offset 分配的真实成本。
4. **本地 mbarrier**:smem mbarrier arrive+wait 一个周期的 cycles 数(clock64 测),TMA 式流水的卡内同步成本参考值。

```bash
./sync_cost
./sync_cost --rounds 5000 --sizes 0,2K,32K --src 0 --dst 1
```

注意:

- ping-pong 和 atomic 内核在自旋,Windows/WDDM 下有 watchdog 风险,`--rounds` 不要设得太大;Linux 服务器(H20)上无此问题。
- `atomicAdd_system` 打 peer 显存需要硬件原生跨设备原子支持(`p2p_ce_vs_tma` 打印的 `native-atomics` 属性)。NVLink(H20)上没问题;PCIe 上该属性通常为 0,atomic 一项的结果可能非法或极慢——这本身也是一个有效结论:PCIe 上别用远端 atomic 做信号。

---

## 建议的完整跑数流程

```bash
# 0. 锁频(可选但强烈建议),记录环境
nvidia-smi -lgc 1500,1500        # 数值按卡型调整
nvidia-smi topo -m

# 1. 裸通信曲线(已有)
./pk_bw_sweep
./p2p_ce_vs_tma

# 2. 同步原语成本
./sync_cost

# 3. 干扰矩阵:源端 + 接收端,再加 starvation
./interference_matrix --csv im_src.csv
./interference_matrix --probe-dev dst --csv im_dst.csv
./interference_matrix --mode starve

# 4. 解锁
nvidia-smi -rgc
```

有了 1-3 的数据,融合成本模型可以写成:
`T_fused ~= max(T_c x S_c(comm, bottleneck), T_m x S_m(comm, bottleneck)) + N_tiles x sync_cost`,
再用真实 fused kernel 验证预测误差。