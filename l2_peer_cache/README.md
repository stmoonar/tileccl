# l2_peer_cache：NVLink P2P 读**不会**进请求方（接收端）L2

一句话结论：GPU A 通过 NVLink/P2P 读 GPU B 的显存，这些数据**不会被 A 自己的
L2 缓存**。读 K 遍 8 MB 远端 buffer，链路上就是 K × 8 MB，一个字节都省不掉。

这个目录不是复述结论，而是**在本机器上把它测出来**，并且带一个物理预期已知
（≈ 有台阶）的对照组——没有对照组的 "peer 曲线是平的" 什么也证明不了，因为
一个坏掉的测量方法产出的曲线也是平的。

---

## 根因：NVIDIA 的 L2 是 memory-side cache

L2 slice 挂在 XBAR **后面**、和 memory partition / 显存控制器成对，不在 core
一侧。一次远端读的路径是：

```
SM → L1 → XBAR → NVLink → (对端 hub) → 对端 XBAR → 对端 L2 → 对端 HBM
                                                    ^^^^^^^
                                          服务这次请求的是「内存归属方」的 L2
```

memory-side cache 按定义只能缓存**它所前置的那块内存**的地址。远端地址根本不
落在本地任何一个 memory partition 上，本地 L2 无从缓存。另一条独立推理：离散
GPU 之间的 L2 **没有硬件一致性协议**，本地缓存一行远端数据在语义上就是不可靠
的，硬件不会这么做。

**关于官方文档：没有一句 NVIDIA 明写的「peer memory 不进 requester L2」。**
官方侧的证据是间接的——从 Fermi 白皮书起，每一代架构图都把 L2 slice 和显存
控制器画在一起。所以这个结论由「架构描述 + 多方独立实测」支撑，不是 spec 里
的一行承诺。需要白纸黑字官方背书的话，没有。

### 独立佐证（本目录之外的）

| 来源 | 说了什么 |
|---|---|
| Lutz et al., *Pump Up the Volume*（SIGMOD 2020） | 最直接：观察到小哈希表在 NVLink 2.0 下不进 GPU L2，并明确给出理由——L2 是 memory-side 的，无法缓存远端数据。同行评议 + 实测 |
| *Spy in the GPU-box*（多 GPU 侧信道，DGX-1/Pascal） | 逆向了跨 GPU 的 cache 组织；整个攻击建立在「本地 GPU 的远端访问会在**远端** GPU 的 L2 上制造 contention」之上。攻击成立本身就是佐证 |
| NVIDIA 开发者论坛实测 | 256 MB 远端 buffer，链路上收到约 268 MB，结论是纯直通读、零本地缓存 |
| GH200 NVLink-C2C | 即使在 cache-coherent 的 C2C 上，用 NCU counter 验证 GPU SM 读 CPU 侧内存（`cudaHostAlloc` / `malloc`，ld/st 或 TMA 都一样）不进 GPU L2，而 HBM 上的数据会进。同一个 memory-side 原理 |

**Blackwell 没有公开实测数据**，也没看到有人专门验证过这一代是否有变化。所以
这个目录默认 `ARCH=sm_90`，但 `make ARCH=sm_120` 一样能跑——sm_120 那对卡是
PCIe P2P，结论方向相同，数值不同。

---

## 三个互相独立的测量

全部用**同一个 kernel、同一张卡**执行，唯一变化的是 buffer 归谁所有：

- `target=local` — buffer 在读卡自己的 HBM 上，**预期有缓存**（对照组）
- `target=peer` — buffer 在对端 HBM 上，**这是被测项**

| mode | 测什么 | `local` 预期（对照） | `peer` 预期（结论） |
|---|---|---|---|
| `bw` | 读带宽 vs 工作集大小，横跨 L2 容量扫 | 工作集 < L2 时跑 cache 带宽，> L2 掉到 HBM 带宽，**有明显台阶** | **全程平**在链路带宽上：1/64 × L2 和 8 × L2 一样快 |
| `lat` | 依赖链 pointer chase（128 B 随机环，单线程）延迟 vs 工作集 | 小工作集 ≈ L2 命中延迟，大工作集 ≈ HBM 延迟 | 全程 ≈ 链路延迟，**不会掉到本地 L2 的量级** |
| `wire` | 直接数字节：读 K 遍，问 NVLink 硬件计数器实际过了多少字节 | ≈ 0（根本没上链路） | **RX/asked ≈ 1.00** |
| `once` | 单次确定性 launch，给外部 profiler 用 | — | — |

`lat` 是那个能区分「被本地缓存了」和「没缓存但被链路带宽卡住」的仪器：如果请求
方 L2 真的持有了这些行，小工作集的 chase 会掉到本地 L2 延迟（10² ns 量级），而
不是停在链路延迟（10³ ns 量级）。带宽平坦有可能是链路饱和造成的错觉，延迟平坦
不会。

### 两个正交的轴：`--via` 和 `--dir`

上面三个 mode 测的是「用 `ld.global` **读**远端」。这只是四个格子里的一个，而且
不是本仓库最关心的那个——`exp3_gemm_rs_fused` 拉远端 tile 用的是 **TMA**，
`tests/p2p_ce_vs_tma` 推远端用的是 **store**。所以 `bw` / `wire` / `once` 三个
mode 都能沿两个轴切换（`lat` 不能：依赖链 chase 没有 TMA 或 store 的对应物）：

| | `--dir read`（pull） | `--dir write`（push） |
|---|---|---|
| `--via ldst` | `ld.global`，默认组合 | `st.global` 打进对端 VA |
| `--via tma` | `cp.async.bulk` peer gmem → smem（**exp3 走的就是这条**） | `cp.async.bulk` smem → peer gmem |

四个格子问的是同一个问题：**发起这次访问的那张卡，它自己的 L2 有没有参与？**
判读方式完全一样——`local` 必须有台阶（仪器有效），`peer` 必须平，链路必须扛走
每一个字节。

两个轴各自带来一点新东西：

- **`--dir write` 没有 L1 干扰。** 全局 store 从 Volta 起就是 write-through 的，
  L1 不持有脏的全局数据，所以写侧曲线上任何台阶都只能是 L2。读侧则必须靠
  `ca` / `cg` 两个 build 才能把 L1 和 L2 分开。
- **`--dir write` 的链路方向是反的。** push 表现为发起方 TX / 归属方 RX，pull 是
  发起方 RX / 归属方 TX。CSV 里 `wire_rx_bytes` 恒为**近端**计数器、
  `wire_tx_bytes` 恒为**远端**计数器，两边照样必须对得上。
- **`--dir write` 的计时含「真的落地了」。** ld/st 写 kernel 结尾有一次
  `__threadfence_system()`，TMA 写 kernel 结尾用 `cp.async.bulk.wait_group 0`
  而不是 `.read` 变体。否则 kernel 可能在 posted write 还在链路上飞的时候就退出，
  测到的是发射率不是交付率。代价是每个 kernel 一次排空，摊到 `iters` 上可忽略。
- **写侧的 verify 问归属方。** 写 kernel 全程写同一个常数（回绕造成的重叠写因此
  无竞争），跑完在**归属方那张卡上**逐字检查——问的是「字节真的落进对端 HBM 了
  吗」，不是「发送方以为自己发出去了吗」。
- **TMA 的回绕在 tile 粒度上**（每 CTA 一个发射线程、4 级流水），ld/st 的回绕在
  16 B 粒度上（每线程）。两者都保证工作集精确等于 `size` 且被完整覆盖，所以两条
  曲线共用一根 x 轴。

**为什么 `bw` 用「回绕」而不是普通 grid-stride**：512 KB 工作集上，500+ 个 CTA
的 grid-stride 会让绝大多数 CTA 一个字节都不读，小工作集端测到的就变成「8 个 SM
打 L2」而不是「整卡打 L2」——而 local 那个台阶正是不能被低估的对照。回绕之后每个
线程在任何尺寸下都做同样多次 load，扫描过程中唯一变化的量就是这些 load 摸到多少
不同的内存。

### 字节计数走 nvidia-smi 而不是 NCU

`nvidia-smi nvlink -gt d` 读的是每条 link 累计的 **data（payload）**计数器，
不需要 root、不需要 profiling 权限。读请求本身不带 payload，所以一次 pull 表现为
**请求方 RX** 和**归属方 TX**，两边必须对得上——这本身就是计数器可信度的自检。
探针在 kernel 循环前后紧贴着采样，并先测一段 idle 漂移做扣除。

NCU 那条路 `run_all.sh` 也会跑（作为独立第二信源），但有个坑：**NCU 的
"peer traffic" 计数器只统计 PCIe 连接的 GPU，不计 NVLink**，必须用
`--section Nvlink`。另外 NCU 需要 profiling 权限，否则 `ERR_NVGPUCTRPERM`。

---

## 构建与运行

```bash
cd l2_peer_cache
make                      # sm_90；make ARCH=sm_120 跑本地 Blackwell PCIe 对
./run_all.sh              # 全套 + 打包
READER=0 OWNER=1 ./run_all.sh     # 选卡（默认 0,1）
QUICK=1 ./run_all.sh              # 短窗口
FORCE=1 ./run_all.sh              # 跳过环境预检的 abort
NO_NCU=1 ./run_all.sh             # 不跑 profiler
```

`make` 出两个二进制，同一份源码，只差 L1 策略：

| binary | 编译 | 用途 |
|---|---|---|
| `peer_l2_probe` | 默认 | 真实 kernel 会遇到的行为 |
| `peer_l2_probe_nol1` | `-Xptxas -dlcm=cg` | 绕过 L1，**L2 结论从这个 build 读** |

单独跑：

```bash
./peer_l2_probe --reader 0 --owner 1                    # 三个 mode 全跑
./peer_l2_probe --modes bw --sizes 1M,8M,64M,512M
./peer_l2_probe --modes wire --wire-sizes 8M --wire-total 64G
./peer_l2_probe --dir write --modes bw,wire             # push 侧（st.global）
./peer_l2_probe --via tma  --modes bw,wire              # pull 侧（cp.async.bulk）
./peer_l2_probe --via tma --dir write --modes bw,wire   # push 侧（TMA）
./peer_l2_probe --help
```

`--via tma` 需要 sm_90+；探针启动时检查 compute capability，不够就直接报错退出，
不会去跑一个空壳 kernel。`--tma-tile`（默认 8K）控制每次 `cp.async.bulk` 的字节
数，默认值让 4 级流水的 staging buffer 停在 48 KiB 静态上限以内。

跑之前锁频（只影响绝对值；结论读的是同一次 sweep 内部的比值，掉频也活得下来）：

```bash
nvidia-smi -lgc <freq> -i 0,1        # 频率用 ../comm_comp/pick_clock.sh 挑
```

### L1 那个 nuance

有博客用 `-Xptxas -dlcm=cg` 做对比，发现关掉 L1 后 NVLink 读性能下降，推断 L1
是参与缓存远端数据的；同一篇也确认 L2 层面数据缓存在**对端** GPU 上。博客级间接
测量，不当定论——所以这里直接把两个 build 都跑了，让 L1 的贡献自己显形。默认扫描
的最小几个尺寸（64K/128K/256K/512K）故意压在 per-SM L1 容量以下，这样 **L1 台阶
和 L2 台阶会落在 x 轴的不同位置**，不会互相冒充。

即便 L1 确实缓存远端行，也不改变 L2 结论，而且对融合 GEMM 意义不大：L1 是
per-SM 的、容量小、CTA 之间不共享，帮不了 GEMM 里 A tile 的跨 CTA 复用；跨设备
可见性通常还要求 `.sys` scope 或 volatile load，本身就绕过 L1。

**4×H800 上的实测结果（`results_20260814_051154`）：L1 确实缓存远端行，而且效果
很大。** 1 MiB 工作集下默认 build 的 `RX/asked = 0.003`——链路字节少了 323 倍；
同一个点在 `-dlcm=cg` 下是 1.000。扣掉冷填成本后 peer 的 L1 命中延迟约 31 ns，
和 local 的 34.7 ns 同一量级，**L1 对远端行和本地行一视同仁**。台阶落在「每 SM
足迹越过 256 KiB」处，正是 Hopper 的 unified L1/SMEM 容量。

这条只对**普通 `ld` + 默认编译**成立。`--via tma` 那一栏就是用来回答「bulk copy
是否同样填 L1」的——如果它是平的，上面这个红利对 `exp3_gemm_rs_fused` 不适用。
另外注意反面：L1 能把一行远端数据攥住到复用几百次，所以用非 volatile 的普通 `ld`
去读对端正在写的 flag，可以拿到任意久的陈旧值。

---

## 输出与判读

`run_all.sh` 产出 `results_<stamp>/` + zip：

```
verdict.txt              ← 先看这个：三个数 + 一句结论
manifest.txt             每一步的命令 / 返回码 / 耗时
env.txt                  git / nvcc / nvidia-smi / topo -m / nvlink -s / 计数器初值
ca_bw.csv  ca_lat.csv  ca_wire.csv      ld/st pull，默认 build
cg_bw.csv  cg_lat.csv  cg_wire.csv      ld/st pull，L1 bypass build
st_bw.csv  st_wire.csv                  ld/st push（只跑 cg：-dlcm 不管 store）
tma_ca_*.csv  tma_cg_*.csv              TMA pull，两个 build
tmast_bw.csv  tmast_wire.csv            TMA push
bigbuf_wire.csv          512 MB（≫ L2，物理上不可能缓存）的定标点
ncu_nvlink_{peer,local}.txt             NCU Nvlink section（ld/st pull）
ncu_metrics_{peer,local}.txt            dram__bytes_read / lts__t_sectors ...
ncu_{nvlink,metrics}_st_{peer,local}.txt      ld/st push
ncu_{nvlink,metrics}_tmald_{peer,local}.txt   TMA pull
ncu_{nvlink,metrics}_tmast_{peer,local}.txt   TMA push
gpu_trace.csv            100 ms 采样的 SM 频率 / 功耗 / 降频原因
load_check.log           负载下频率是否守得住
```

`_bw.csv` / `_wire.csv` 末尾追加了 `via,dir,tma_tile_bytes` 三列（**追加不插入**，
因为 verdict 的 awk 按列号读，旧 bundle 也得继续能读）。

`verdict.txt` 只认三个数（都取自 `cg` build）：

```
CONTROL  local L2 knee            = 2.47x  [ok: the instrument sees an L2]
TEST     peer  L2 knee            = 1.01x  [flat: nothing cached locally]
TEST     peer  wire RX / asked    = 1.009  [every byte crossed the link]

=> PEER MEMORY IS NOT CACHED IN THE REQUESTER-SIDE L2.
```

**如果 CONTROL 那行不成立（local 没有台阶），脚本会直接给 INCONCLUSIVE 而不是
给结论**——仪器都没证明自己能看见 L2，peer 平坦就没有信息量。

verdict 里另外三块：

```
L1       peer wire RX/asked at 1 MiB: default 0.003 vs -dlcm=cg 1.000  [L1 DOES cache peer lines]

---- PUSH, ld/st: st.global into peer memory ----
  bw  local  in-L2 ... knee 2.xx     bw  peer ... knee 1.00
  L1? peer 64K vs 100M =  0.93x  [flat: nothing cached per-SM]
  wire peer  1.0 MiB -> near/asked 1.000, far/asked 1.000
---- PULL, TMA: cp.async.bulk peer gmem -> smem (default L1) ----
---- PUSH, TMA: cp.async.bulk smem -> peer gmem ----
```

- **`L1` 那行**是刻意补上的。上面所有 L2 判据都只取 `frac_l2 ≥ 0.25` 的点——正是
  为了不让 L1 污染 L2 结论，代价是它们**看不见 L1**。这一行看得见：同一个小工作
  集，默认 build 对 `-dlcm=cg`。
- **`L1?` 那行**对每条路径各算一次（最小工作集 vs 远超 L2 的点）。它是判断
  「ld.global 上观察到的 per-SM 缓存，在这条路径上还成不成立」的那个数。TMA pull
  那一栏如果是 `[flat]`，说明 bulk copy 不填 L1，读侧那个 L1 红利对
  `exp3_gemm_rs_fused` **不适用**。
- 每条路径都自带 `local` 对照，可以独立判读，不依赖别的格子。

画图：

```bash
python plot_peer_l2.py results_<stamp>        # 写到 results_<stamp>/figs/
```

三联图：(a) 带宽 vs 工作集（local 有台阶 / peer 平），(b) 延迟 vs 工作集，
(c) 链路字节 / 请求字节。实线 = 默认 build，虚线（柱状图里是斜纹）= L1 bypass。

NCU 那侧的读法：`target=local` 时 `dram__bytes_read.sum` ≈ 8 MiB（只有第一遍冷
读，其余全是 L2 命中）；`target=peer` 时它应该 ≈ 0，而 Nvlink section 显示收到了
整整 `--total` 那么多字节。

---

## 已知的坑

- **计数器可能是 N/A。** 某些驱动 / vGPU 下 `nvidia-smi nvlink -gt d` 不给数。
  探针会打 `[!] ... counters unavailable` 并继续；带宽和延迟证据不受影响。
- **`wire` 的分母是逻辑读取量，不是工作集。** 小工作集时几百个 CTA 同时读同一
  批 line，如果哪一级做了请求合并，RX/asked 会明显小于 1。这不是「L2 缓存了」，
  是合并；`bigbuf_wire.csv`（512 MB，缓存物理上不可能发生）就是用来定标这件事的
  ——两个点都回到 1.00，小 buffer 那个 1.00 才不是计数器假象。
- **`lat` 的大工作集点还要付 TLB miss。** 绝对值别单独引用，读 local/peer 的
  **对比**。
- **没有直接测「对端 L2 确实缓存了」。** 那需要在归属方观测 DRAM 流量，而 NCU
  无法把对端 kernel 之外的 peer 请求归因到某个 kernel 上。本目录只证明了「不在
  请求方」；「在归属方」目前靠架构论证 + 上表的第三方证据 + `tests/ANALYSIS.md`
  第 6 节的端到端行为证据（H20 上融合方法击穿模型下界 5–40%，因为消费者读到的是
  刚推过来、还热在 **dst L2** 里的 tile）。
- **`--via tma` 只测了 1D 的 `cp.async.bulk`，没测 tensor-map 变体**
  （`cp.async.bulk.tensor`，CUTLASS 的 `SM90_TMA_LOAD` 实际用的那个）。两者共用
  同一个 TMA 引擎和同一条 gmem 通路，地址路由不会因为多了一层 tensor map 描述符
  而变化，但严格说这是推断。真要较真，得把 descriptor 建在 peer 指针上再跑一遍。
- **`--via tma` 的 `local` 对照跑的是 TMA，不是 ld.global**，所以两条 via 之间的
  绝对带宽不可直接比（发射方式、流水深度都不同）。要比的是每条 via 内部
  local/peer 的**比值**。
- **绝对带宽跨 bundle 不可比**（本仓库通例）：不同锁频下的数字不要放在一起。

---

## 和本仓库其他实验的关系

这是 `comm_comp/exp2_ag_gemm_granularity` 和 `exp3_*` 背后的一条前提。既然远端
数据不进本地 L2，那么融合 GEMM 里「A tile 从对端直接读、指望 L2 帮忙做跨 CTA
复用」这条路根本不存在：每个 CTA 的每次访问都是一次实打实的链路往返。这正是
flux 那类设计必须先把远端数据**显式搬进本地（HBM 或 smem）**、再让计算复用它的
原因，也是 `exp2` 里 comm 侧对 chunk 粒度不敏感、代价全在 compute 侧 tile
quantization 的背景。
