# P2P microbench

四个独立的可执行文件，`make` 一次全部构建：

| binary | 回答的问题 |
|---|---|
| `p2p_ce_vs_tma` | 通算融合应该走哪条搬运路径、tile 该多大（按 token 粒度扫） |
| `pk_bw_sweep` | 复现 ParallelKittens 论文（arXiv:2511.13940）Fig.2 / Fig.3 / Tab.1：不同传输机制在不同**消息大小**和**SM 数**下打到的带宽 |
| `interference_matrix` | 通信对计算、计算对通信的**双向干扰矩阵**(通信方式 x 计算瓶颈类型),以及满载 SM 下三种搬运方式的 progress/starvation 行为 |
| `sync_cost` | 细粒度 pipeline 的同步原语成本:跨卡 flag 单向延迟、fence+signal 尾部开销、远端 atomic 往返、本地 mbarrier 周期 |
| `intra_sm_matrix` | **SM 内部**(warp-specialized 融合)的通算干扰:通信 warp 和计算 warp 同 CTA 时,issue slot / LSU / 内存管线的动态竞争与让出 warp 的静态代价 |

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

CTA 布局:probe 与 comm kernel 同卡时,probe 拿 (SM 数 - `--comm-sms`) 个 CTA;`ce` 行 probe 拿全部 SM——这正是 CE 的卖点。注意 `S_c` 的 alone 基线用与 overlap **相同的 CTA 数**测得,所以 `S_c` 只反映 memory/pipe 竞争;把 CTA 让给通信的机会成本单独体现在 `ce` 行与 `tma`/`reg` 行 alone 列的差值上(compute-bound 探针上应近似 comm-sms/SM 数的线性损失,HBM-bound 探针上可能近似为零甚至为负)。

TMA 消息上限:消费卡 smem 远小于 H100 的 227 KB,`--msg` 装不下 ≥3 级流水时程序自动把 tma 的消息折半到能装下为止,并打印实际值;tma 行的带宽按实际搬运字节数计算。

### 两种模式

- `--mode matrix`(默认):上述干扰矩阵。
- `--mode starve`:实验 C。一个满占用的 ffma kernel 常驻源卡(CTA 数由 occupancy API 算出,占满每个 SM 的全部 CTA slot——只占部分 slot 的话 comm kernel 仍能 co-schedule,测不到饿死),再让每种搬运方式发一次 payload。预期:`ce` 照常完成;`tma`/`reg` 拿不到 CTA slot,饿死到 squatter 退出为止。这是三者 progress model 的本质差别,带宽曲线看不出来。

```bash
./interference_matrix                                  # 全矩阵
./interference_matrix --probes hbm,mma --comms ce,tma
./interference_matrix --probe-dev dst                  # 接收端干扰(实验 E)
./interference_matrix --mode starve
./interference_matrix --csv results.csv                # 机器可读输出
```

主要参数:`--total`(每次通信迭代的 payload,默认 64M)、`--msg`(消息大小,默认 2M,须整除 total)、`--comm-sms`(tma/reg 的 CTA 数,默认 8)、`--probe-sms`(0 = 自动)、`--window-ms`(每格测量窗口,默认 300)、`--probe-dev src|dst`、`--hbm-buf`(hbm 探针工作集,默认 256M)。

### 结果解读与前提

1. **锁频最好,锁不了也有办法**:`nvidia-smi -lgc` 需要宿主机层面的权限(容器内 root 通常不够)。不锁频时用三件事替代:(a) 看内置 sanity 行——`ce` x `mma`/`ffma`/`l2`/`smem` 理应是 1.00/1.00,降频会不分行地拖慢一切,这几行不是 1.00 就说明该 run 被 DVFS 污染;(b) 旁路采样 `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader -lms 500 > clocks.log` 留档;(c) 加 `--no-alone-cache`,让每格的 alone 基线与 overlap 背靠背相邻测量,抵消热漂移(run 时间约翻倍)。
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
---

## intra_sm_matrix:SM 内部(warp-specialized 融合)的通算干扰

`interference_matrix` 测的是 inter-SM 干扰(通信是独立 kernel,与计算抢 DRAM/L2/链路和 CTA 名额)。真正的融合 kernel 里,通信 warp 和计算 warp 在**同一个 CTA**,抢的是另一组资源:warp scheduler 的 issue slot、LSU/MIO 管线、寄存器文件、smem 容量与 bank。这个工具量化的就是这部分。

### 方法学

一个 fused kernel,每 SM 一个 CTA、每 CTA W 个 warp(默认 16):

- warp [0, C):通信 —— `ldst`(uint4 直写 peer VA)或 `tma`(lane 0 驱动 cp.async.bulk 流水,local gmem → smem → peer gmem);
- warp [C, W):计算探针 —— `mma` / `ffma` / `smem` / `hbm` 的 per-warp 版本。

全部跑到 stop flag,每个 warp 上报迭代数,同一窗口内同时得到计算吞吐和通信带宽。**两个基线把"给通信让 warp"的代价拆成两半**:

- `S_warp` = R(C=0 全算) / R(通信 warp 进场即退出):让出 C 个 warp 的**静态代价**(纯并行度损失,无竞争);
- `S_intra` = R(通信 warp 退出) / R(通信 warp 活跃):同样 warp 数下,通信活动本身的**动态代价**(issue slot、LSU/MIO、内存管线)。

预期模式(校验数据用):`tma` 的 S_intra 应接近 1.00(每 warp 只有一个线程偶尔发 bulk 指令);`ldst` 对 `ffma`(issue-bound)和 `smem`(LSU/MIO-bound)的 S_intra 应明显大于对 `mma`(tensor pipe 独立,发射率低);`S_warp` 对 compute-bound 探针应 ≈ W/(W−C)。

```bash
./intra_sm_matrix                                      # 全矩阵,cw 扫 1,2,4
./intra_sm_matrix --probes ffma,smem --comms ldst --comm-warps 1,2,4,8
./intra_sm_matrix --warps 16 --msg 16K --csv intra.csv
```

主要参数:`--warps`(每 CTA warp 数,默认 16)、`--comm-warps`(通信 warp 数扫描点)、`--msg`(消息大小,tma 会按 smem 预算自动折半)、`--stages`(tma 流水深度)、`--blocks`(默认每 SM 一个 CTA)。

### 注意

1. CE 天然没有这张表——它不可能出现在 CTA 里,intra-SM 干扰恒为零;这张表量化的正是 SM 驻留方案相对 CE 多付的那部分。
2. 每 SM 只有一个 CTA(模拟 FlashAttention 风格的融合 kernel),计算探针的绝对吞吐低于 `interference_matrix` 里满占用的版本,横向对比只看比值。
3. tma 的 staging smem 会挤占融合 kernel 的 smem 预算,这个静态代价体现在打印出的 dyn smem 数字上,不在 S_intra 里。

### v2 修正(第一轮数据暴露的问题)

第一轮 RTX5000 数据出现了"纯寄存器的 mma 探针被 1 个 tma warp 拖慢 5 倍,且对消息大小敏感"的异常,定位为两个混杂因素,已修:

1. **测量循环污染**:探针每次迭代做一次 stop-flag load + publish store,在 P2P backpressure 顶满 SM 访存端口时,这两条测量用访存指令把纯计算探针变成被访存延迟 gate 的循环。修复:寄存器探针每 `kCheckBatch=16` 次迭代才碰一次内存。
2. **极端 backpressure 配置**:默认所有 CTA 都带通信 warp,数百个 warp 分一条链路,每个通信 warp 大部分时间在 stall/spin,测到的是"堵死的通信 warp 有多吵"。修复:tma 自旋加 `__nanosleep`(礼貌自旋,生产 kernel 的标准做法);新增 `--comm-ctas N` 只让前 N 个 CTA 带通信 warp,其余纯计算 CTA 单独报告 `S_pure` 列——它同时量化了"通信 CTA 的 backpressure 会不会溢出到没有通信 warp 的 SM"(经由共享的 L2/内存端口)。

注意:backpressure 拥塞 SM 访存端口、连累同 SM 所有 warp 访存延迟,这是 intra-SM 特有的真实干扰通道(计算 warp 必然访存,躲不开);v2 只是不再让它被错误记到纯寄存器探针头上。对比 `--comm-ctas 8` 与默认全 CTA 两种跑法的 S_intra,可以分离"稳态资源共享"与"backpressure 拥塞"两种成分。

---

## pipeline_e2e:实验 D,端到端验证成本模型

前面所有工具产出的都是**成本模型的参数**;这个工具验证**模型本身**:一个合成 producer → transfer → consumer 融合 pipeline 跨两卡跑,对每个 (方法, 算术强度) 组合测四个量,并当场对比理想稳态模型的预测:

- `Tc_src`:只生产(同样的计算,tile 写到**本地**);
- `Tm`:只传输(intensity=0,只有 seed + store);
- `Tc_dst`:只消费(flag 预置、buffer 预填);
- `To`:全流水的双卡 wall clock;
- `pred = max(Tc_src, Tm, Tc_dst)`,`err% = (To − pred) / pred`。

`err%` 就是模型没解释掉的时间——干扰、流水 fill/drain、同步开销。预期签名:两端(强 comm-bound / 强 compute-bound)err 小,交叉点(Tc ≈ Tm)附近有一个 bump。err 大且系统性偏正的格子,说明干扰矩阵里对应的 S 因子必须进模型。

三种传输方法对应"产出的 tile 在哪"的三条通路,消费者对所有方法完全相同(acquire-poll flag → FMA 链 → sink):

| method | 生产侧路径 | 同步 |
|---|---|---|
| `ce` | register → local gmem staging → 分 chunk `cudaMemcpyPeerAsync` | stream/event 依赖链 |
| `ldst` | register → **直写 peer gmem**,零本地 staging | 每 tile 一个 `st.release.sys` flag |
| `tma` | register → smem(双缓冲)→ `cp.async.bulk` → peer gmem | flag 在 bulk group 完成后释放(滞后一个 tile) |

校验:每个 tile 的 element 0 由消费者按生产者的公式重算比对,`verify` 列报告。

```bash
./pipeline_e2e                                          # 三方法 x 强度 0..4096
./pipeline_e2e --methods ldst,tma --intensity 0,256,1024,4096
./pipeline_e2e --total 128M --tile 16K --csv e2e.csv
```

主要参数:`--intensity`(每 float 的 FMA 数,FLOP/byte = intensity/2;交叉点位置 ≈ 计算吞吐/链路带宽,PCIe 上在 1000+,NVLink 上会低得多)、`--tile`(默认 16K,tma 需要 2×tile 装进 smem)、`--chunks`(ce 流水段数,默认 16)、`--reps`(取最小值,默认 3)。

读法:先看 `verify` 全 ok(说明 flag 语义和传输路径正确);再看每列 To 随强度的走向——低强度贴 Tm、高强度贴 Tc,交叉点两侧谁的 To 低,谁就是该强度区间的正确融合方式;最后看 err% 的分布,决定成本模型要不要加干扰修正项。

---

## plot_interference.py:干扰与同步开销图集

与 `plot_pk_bw.py` 同风格:两台机器的实测数据内嵌在脚本里,重跑工具后把新数字贴回去即可刷新。`python plot_interference.py --out figs` 生成六张图:

| 图 | 内容 |
|---|---|
| `if_sc_heatmap.png` | S_c 热图(2 平台 x 收发两侧):通信把计算拖慢多少。全图唯一的大格子是 NVLink 的 L2 污染 |
| `if_sm_heatmap.png` | S_m 热图(发送侧):计算把通信拖慢多少。PCIe 上 CE 最脆,NVLink 上欠配置的 SM 系最脆 |
| `if_sm_sweep.png` | S_m 与 alone 带宽随通信 SM 数的变化:健壮性 = bytes-in-flight,配置到饱和点即免疫 |
| `if_intra.png` | intra-SM:S_intra(全员 vs 集中两种形态)+ S_warp(静态让 warp 代价) |
| `if_sync.png` | 同步原语:producer 序列成本、release flag vs fence 的边际开销、单向延迟/远端 atomic 参考线 |
| `if_starve.png` | starvation:满占用计算 kernel 下只有 CE 能独立进展 |
