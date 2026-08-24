# comm_comp — 通信/计算干扰实验（4×H800 NVLink, sm90）

一组独立的 CUDA/CUTLASS 微基准，量化通信与计算在 Hopper 上的资源竞争，
实验设计参考 [flux](https://github.com/bytedance/flux) 的 **sm90** AG+GEMM 与
GEMM+RS 实现（不是 sm80 路径）。全部单进程多卡：rank r == CUDA device r，
P2P 走 `cudaDeviceEnablePeerAccess` + UVA，不需要 MPI / NCCL / nvshmem /
pybind11。

GEMM 统一为 CUTLASS 3.x-API 的 sm90 fp16 GEMM（fp32 累加）：
`KernelTmaWarpSpecializedCooperative` + `TmaWarpSpecializedCooperative`
epilogue，tile 128×256×64，cluster 2×1×1 —— 与 flux sm90 dense 路径同一族
配置（flux 的 H800 调优表也在 128×{128,256}×64 一带）。

## 编译

```bash
git submodule update --init          # 需要 third_party/cutlass (v4.6)
cd comm_comp
make -j3                             # nvcc -arch=sm_90a，CUDA >= 12.4，12.9 已验证 API
```

跑分前先锁频，否则 DVFS 会伪装成干扰：

```bash
nvidia-smi -lgc 1830 -i 0,1,2,3      # 已有的 4×H800 结果包全部锁在 1830
```

> **锁 1830，不是上限 1980。** `reports/` 下所有 4×H800 结果包（20260810 /
> 20260811）都是 1830 跑的；换频率跑出来的绝对时间与它们不可比。

## 一键跑全部实验 — `run_all.sh`

```bash
nvidia-smi -lgc 1830 -i 0,1,2,3        # 先锁频（脚本只记录时钟，不改）
./run_all.sh                            # 全量，约 20-40 分钟
QUICK=1 ./run_all.sh                    # 快速版
FORCE=1 ./run_all.sh                    # 跳过环境前置检查（结果按脏数据对待）
```

**环境前置检查**：开跑前用 `nvidia-smi` 检查目标 GPU（尊重
`CUDA_VISIBLE_DEVICES`）：利用率 >5%、显存被其它进程占 >2 GiB、或 SM 时
钟 <1000 MHz（未锁频的空闲卡会掉到 ~345 MHz；锁频后即使空闲也保持锁定
值）任一命中就拒跑。共享机上跑出来的数字比没有数字更糟——它们看起来是真
的。注意程序里打印的 "1980 MHz" 来自 `cudaDevAttrClockRate`，是**峰值**
而非实时频率；判断是否锁频以 `env.txt` 里 `clocks.sm` 的当前值为准。

编译 + 依次跑完全部实验（含 `--verify`），日志/CSV/环境快照（nvidia-smi
拓扑、时钟、nvcc 版本、git commit）收进 `results_<时间戳>/` 并打包成
zip（机器上没有 zip 则 tar.gz）。每个实验都有 timeout，某一项挂死（记为
rc=124）或失败不影响其余项，结果见包内 `manifest.txt`；fused 实验超时后
会自动清理残留的 rank 进程。把 zip 拿回来即可分析。

## 实验一：CE 搬运对独立 GEMM 的影响 — `exp1_ce_interference`

GEMM 在 `--gemm-dev` 上跑，同时 CE 在各 rank 之间搬运与 GEMM **无关**的数
据（对应 flux AG+GEMM 里 `cudaMemcpyAsync` 走 copy engine 的流量形态）。对
照组 = 无 CE 流量的同一 GEMM。输出 `S_c = t_overlap / t_alone` 与两组
TFLOP/s。

流量模式用于把"本地 HBM 带宽竞争 / CE 引擎本身 / NVSwitch 负载"三个因素
分离：

| pattern | 含义 | 预期 |
|---|---|---|
| `pull` | 本卡 CE 从各 peer 读，写本地 HBM（flux AG pull 模式） | 主要效应 |
| `push` | 各 peer 的 CE 写入本卡 HBM（flux AG push 模式） | 主要效应 |
| `allgather` | 所有 rank 互拉，完整 AG 环境流量 | 最接近真实 |
| `bystander` | 只有其它三张卡互传，本卡不参与 | ≈1.00（对照） |
| `engine-only` | 本卡 CE 在两张其它卡之间搬运，不碰本地 HBM | ≈1.00 则说明 CE 引擎本身免费 |
| `local` | 本卡 D2D，CE 读写均在本地 HBM | 每字节代价上界 |

```bash
./exp1_ce_interference                                    # 默认全模式
./exp1_ce_interference --sizes 8192 --msgs 4M,16M,64M     # 消息粒度扫描（无依赖场景）
./exp1_ce_interference --patterns pull,bystander --csv exp1.csv
```

宿主线程用事件环（ring of bursts）持续喂 CE，保证整个 GEMM 计时窗内 CE
不空闲；若观察到空闲会在行尾标 `[!]`，此时 S_c 只是下界。`ovl GB/s` 只统
计 GEMM 结束前已轮询到完成的 chunk（drain 阶段不计入），是重叠带宽的下界。

每个 shape 的所有模式跑完后会**复测一次 alone 基线**并打印漂移百分比
（`baseline recheck`）：所有 S_c 都是对开头基线的比值，若期间环境变化
（别的任务上机、时钟下滑），漂移 >5% 会标 `[!]`，该 shape 的全部 S_c 应
视为不可信。

## 实验二：搬运粒度 vs 计算效率 — `exp2_ag_gemm_granularity`

真实数据依赖的 AG+GEMM 流水线：gathered A 按 `world × S` 行块切分
（S = 每个远端 shard 的 chunk 数，即扫描轴），comm stream 逐 chunk
`cudaMemcpyPeerAsync` 拉取 + 记录事件，compute stream 对应分段 GEMM 在
`cudaStreamWaitEvent` 之后发射；本地 shard 不需要拷贝、最先计算（与 flux
的 tile-scheduler rank 旋转同义）。comm stream 与 flux 一致用最高优先级。

与 flux sm90 的差异要点：flux 是**单个 persistent GEMM 内核在 tile 粒度自
旋等待 system-scope flag**（fork 了 `GemmUniversal`），拷贝粒度固定为整个
per-rank shard（`SPLIT == 1`）；这里用"分段发射 + 事件依赖"的等价形态跑在
未改动的 CUTLASS 上，`S=1` 即 flux 的默认粒度，S>1 探索更细的重叠。每
chunk 的信号成本（event record + stream wait + launch）本身就是被测的粒度
开销之一。

每个 S 输出四个分量：

- `t_full`：不分段 GEMM（计算 roofline）
- `t_seg` / `seg_eff`：只分段不通信 → 纯 tile 量化损失
- `t_comm` / `GB/s`：只通信 → 该消息尺寸下的 CE 效率
- `t_ovl` / `ovl_eff` / `bub%`：流水线实测（迭代间加闸门，测的是单次
  AG+GEMM 延迟而非稳态吞吐）

```bash
./exp2_ag_gemm_granularity --verify                       # 先验证依赖正确性
./exp2_ag_gemm_granularity --sizes 8192,16384x8192x8192 --chunks 1,2,4,8,16,32
./exp2_ag_gemm_granularity --comm-streams 1               # 串行 ring 顺序拷贝
./exp2_ag_gemm_granularity --push                         # peer 侧 CE 推送
```

`--verify` 会先把 A_full 的远端区域抹脏再跑流水线，与预 gather 的参考结果
逐位对比 —— 事件依赖若有错会立即 FAIL。

## 实验三：epilogue 远端写对计算的影响 — `exp3_epilogue_remote`

唯一通信与计算竞争 SM 资源的场景：GEMM epilogue 直接把 D 写到远端。值得注
意的是 flux 在 sm90 上**刻意不做 epilogue 远端写**（GEMM+RS 的 epilogue
TMA-store 到本地 + 置 flag，由额外 producer warp 从 peer TMA **拉取**再本
地 `red.global.add`），而 sm80 路径是 epilogue 直接向量化 store 写 peer 指
针。本实验直接量化这个设计选择在 Hopper 上值多少。

同一 GEMM，只换 D 指针与 epilogue 形态：

- 模式：`local`（基线）、`remote`（D 整体在 `--peer` 卡上）、
  `scatter-local`（按 world 分段但全写本地，剥离分段本身的开销）、
  `scatter`（GEMM+RS 的写模式：第 r 段写 rank r 的 buffer，1/world 本地）
- epilogue：`tma`（Sm90 TMA store，描述符建在 peer UVA 指针上）与
  `nosmem`（DefaultEpilogue，寄存器按 accumulator 布局直接 st.global，无
  smem 重排 —— 近似 sm80 远端写风格搬到 sm90 硬件。注意这条路径的 store
  不是 128-bit 向量化的，绝对 TFLOP/s 天然低于 tma；有效读法是**同一
  epilogue 内部** remote vs local 的比值，而非 tma 与 nosmem 之间的绝对
  值对比）

默认包含小 K 形状（8192×8192×{8192,2048,512}）：算术强度越低，epilogue 占
比越大，远端写的代价越显。

```bash
./exp3_epilogue_remote --verify
./exp3_epilogue_remote --sizes 8192x8192x512 --epi nosmem --modes local,remote
```

读法：`remote/local ≈ 1.0` 说明 NVLink 写不打扰计算、flux 的 pull 式 RS 在
延迟上没占到便宜；`>> 1.0` 则 pull 设计在 Hopper 上有据。`tma` 行如果在
remote 下报错或塌速而 `nosmem` 正常，说明问题特定于 TMA 描述符 + peer 内
存（flux 全程没有对远端做过 TMA store，此组合属于未踩过的路径 —— 这正是
保留 `nosmem` 变体的原因）。

## 实验三 v2：flux sm90 融合 GEMM+RS — `exp3_gemm_rs_fused`

上面的 `exp3_epilogue_remote` 测的是"epilogue 直接远端写"（flux sm80 风
格）；本实验把 **flux sm90 真实采用的设计**原样搬到 stock CUTLASS 4.6 上
测：epilogue TMA-store 到**本地** + 逐 tile 置 system-scope flag，每个
CTA 的 producer warpgroup 里两个空闲 warp 变成 RS Fetch / RS Reduce
warp——Fetch 自旋等 peer 的 tile flag 然后 TMA 从 peer 拉到 smem，Reduce
从 smem 读出后用 `red.global.add.noftz.v8.f16` 归约进本地 reduce buffer。
通信完全在 GEMM kernel 内部，代价直接体现为 kernel 时间膨胀：

三组对照（一次运行全部测完，同环境）：

```
baseline : 同配置 stock CUTLASS GEMM
ctrl     : 同一个 RS kernel，运行时关闭通信（fetch/reduce warp 空转、
           不发布 flag）——静态调度器 / XOR swizzle / smem carveout /
           mainloop stage 数与 fused 完全一致
fused    : 完整 GEMM+RS

struct = t_ctrl / t_base     kernel 结构改造本身的代价
comm   = t_fused / t_ctrl    通信本身的代价（flag 同步 + NVLink 拉取 +
                             fp16 归约 + 等最慢 peer 的尾部）
total  = t_fused / t_base  = struct × comm
```

没有 ctrl 组时 struct 与 comm 混在一起无法归因（stock baseline 和 fused
的调度器/stage 数本来就不同）；有了它，"融合的净代价"才能干净地归到通信
头上。

移植说明（对照 flux 源码）：

- `rs_gemm_kernel_sm90.cuh`：CUTLASS 4.6 的 cooperative kernel 原文拷贝 +
  重放 flux 的修改。4.6 里静态 persistent scheduler 下 CLC 调度 warp
  （Warp1）和 MainloopAux warp 本来就空转，正好承载 flux 的两个 RS 角色。
- flag 发布在 consumer warpgroup `store()` 之后完成（`tma_store_wait<0>`
  + named barrier + system CAS 0→1），与 flux 放在 EVT AuxStore `end()`
  里等价（同一批线程、同一同步序列），省掉了自定义 EVT 节点；比 flux 多
  一次 named-barrier sync（保证所有 store 线程 retire 后才置 flag）。
- tile 顺序用 flux `WorkIdxMSwizzler` 的 XOR swizzle（nnodes=1 折叠）：
  rank r 在第 t 步生产 segment `s⊕r`，恰好是 rank `s⊕r` 第 t 步要拉取的
  tile——生产与消费天然锁步。要求 world 为 2 的幂。
- flux 强制 epilogue `StagesD=1`（省 smem 换 mainloop stage），这里
  baseline/ctrl/fused 统一保留 builder 默认值（程序会打印 StagesC/StagesD
  与两侧 mainloop stage 数）：ctrl 与 fused 完全同构，这个偏离 flux 的选
  择被 ctrl 列吸收，不再污染 comm 的归因；逐 tile `tma_store_wait<0>` 已
  起到 flux StagesD=1 的排空作用。

**多进程**：一进程一卡（`fork`），buffer 用 cudaIPC 交换（匿名共享 mmap
传 handle），flux 式 device barrier-all 对齐各 rank 的每次迭代——与
torchrun 部署形态一致，不需要 MPI/pybind11。四个 rank 对称执行：每张卡
一边算自己的 GEMM 一边被其它三张卡拉取。仅支持 Linux。

```bash
make exp3_gemm_rs_fused
./exp3_gemm_rs_fused --verify              # 先验证 RS 结果（fp32 参考 + 容差）
./exp3_gemm_rs_fused --m 8192 --n 8192 --k 2048 --csv rs.csv
```

读法：看 `comm` 列（fused/ctrl）——这是融合通信的净代价；`struct` 列
（ctrl/base）是 kernel 结构税，与通信无关。分布式指标看 **worst-rank**
（程序末尾打印），单 rank 均值会低估同步尾部。与 `exp3_epilogue_remote`
的 remote/local 比值对照，可以回答"flux 在 sm90 上选 pull 而不是
epilogue 远端写，赚了还是亏了"。

## 实验四：tile 粒度 AG 融合的传输通道 — `exp4_ag_tile_transport`

flux sm90 AG+GEMM 融合的是**等待逻辑**：CE 搬运 + `cuStreamWriteValue`
发旗标 + GEMM kernel 内自旋（论文 §3.2/§4.3），但论文从未比较过"谁来搬"
—— copy engine（不占 SM、每次拷贝有 µs 级发起开销、只能搬连续地址）vs
SM/TMA 驱动（牺牲 N_comm 个 SM，换来搬运路径上的布局变换和天然的细粒度
旗标）。本实验在 tile 粒度融合下量化这条轴：**每次搬运的数据量变化时，两
种通道对计算时间线各有什么影响**。

4 卡全 AG 对称运行，单进程 UVA。计算是合成的门控负载（真实向量化加载 +
`--intensity` 定容 FMA 链，计算性能本身不是指标），按 (行块, K-slice) 为
单元自旋等待所属 chunk 的旗标（单调 epoch 协议，本地 chunk 预置常真值，
与 flux "本地分片旗标预置"一致）；**两变体的消费者指令流逐条一致**（沿用
tests/pipeline_e2e.cu 的原则），只有旗标生产者和 A 源指针不同：

- `ce`：本 rank 最高优先级 comm stream 上 `cudaMemcpyPeerAsync` 拉取
  G 个行块（连续行，字节数=依赖数据量），随后同 stream
  `cuStreamWriteValue32` 发旗标（flux 原方案；经 `cudaGetDriverEntryPoint`
  运行时解析，无 -lcuda）。CE 做不出 tile-blocked 布局，所以这是**模拟**：
  搬的字节是真实行数据，但消费者从预先变换好的本地 shadow 缓冲加载。
- `tma`：同一 kernel 里 blockIdx < n_comm 的 block 专职通信——对 peer 分片
  的 2D tensor-map TMA load（描述符建在 peer 指针上）→ smem → 1D bulk
  store 成本地 tile-blocked panel → `st.release.sys` 发旗标。消费者读的就
  是真搬来的数据。grid ≤ SM 数保证全部 block 常驻，构造性避免 flux SM-AG
  需要的"producer 已驻留"信号。

扫描轴：`--g`（每次搬运聚合的行块数，粒度主轴）、`--k`（单行块
128·K·2B ≈ 0.25–2 MiB；每 tile 字节与 FLOP 都 ∝K，通信/计算比不随 K 变，
K 干净地暴露 CE 每拷贝固定开销的摊销）、`--n-comm`（仅 TMA）。

每配置的模式行（对照组设计）：

| mode | 含义 | 预期 |
|---|---|---|
| `compute-only` | 旗标全预置；配对基线，fused 后复测漂移 | 分母 |
| `fused` | 真实依赖 + 真实搬运 | 主行 |
| `comm-only` | 只搬运：CE 墙钟（rank=-1 聚合行）/ TMA 事件计时 | 带宽 |
| `bystander` | 搬运照跑但门控直通 → 纯带宽/引擎干扰分量（挂钩 exp1） | 干扰 |
| `local` | 同卡搬运 + 真实门控 | 诊断本地 HBM + 门控成本，不预设 ≈1 |
| `memop-cost` | 仅 CE：纯 write-value 链 + 真实门控 | 诊断 flag 提交成本，不预设 ≈1 |
| `arrival` | 搬运 + observer kernel 记每 chunk 到达时刻（无计算干扰） | 到达曲线 |

描述指标：`slowdown = fused/compute-only`，`interference_sd =
bystander/compute-only`，`stall_sd = slowdown/interference_sd`。最后一个比值
只用于描述，不能当作可加的因果分解。fused 行自带 remote 单元等待时间的
p50/p95/max（kernel 内 `%globaltimer` 时间戳，只做同卡差值）。

新结果中 `t_us_*` 是 consumer kernel 的 active 时间（含设备端自旋，不含
kernel 提交前的宿主空档）；`e2e_us_mean` 是相邻迭代完成点之间的端到端时间，
`host_enqueue_ms` 单列宿主提交/反压。三者必须同时报告，不能把宿主提交慢
直接归因成 HBM 干扰。

```bash
make exp4_ag_tile_transport
./exp4_ag_tile_transport --k 4096 --g 1,4 --n-comm 4 --iters 10 --verify
./exp4_ag_tile_transport --k 1024,8192 --g 1,4,16 --n-comm 8 \
    --csv exp4.csv --dump-tiles exp4_tiles      # per-tile 时间线 + 到达曲线
./exp4_ag_tile_transport --k 8192 --g 4 --n-comm 1,2,4,8,16 --variants tma
```

`--verify` 五件套：(a) 布局变换 vs host 独立参照；(b) TMA staging 与
shadow 逐位一致（transform kernel 与 tensor-map 两条独立路径）；(c) CE
blob 与 peer 分片逐位一致；(d) CE/TMA 两变体计算校验和逐位相等（可交换
整数 wrapping add，网格划分无关）；(e) 消费者读到的 epoch 断言
（`err_count` 列）。CSV 每行带 `flag_mech`（memop/kernel fallback，绝不
静默混行）与 `ce_dst`（fixed/`--ce-cycle-dst`）。

### Exp4 补充：16 KiB panel 粒度 CE/TMA 对照

`--panel-h` 切换到物理 panel 实验。最小单元固定为 A 的
`128×64×fp16 = 16 KiB`；H 表示多少个连续 panel 共用一个 ready flag。
固定 K 时，总 AG 字节数与总合成计算量不随 H 改变：

- `ce-aggregate`：每个 chunk 发一次 `H×16 KiB` 连续 dummy copy，再发 flag；
- `ce-panelized`：每个 chunk 连续发 H 次 16 KiB dummy copy，再发 flag；
- `tma`：H 次真实 `128×64` peer TMA load → local HBM store，再发 flag，
  consumer 读取真正搬到的 panel。

三者的字节数、flag 粒度和 consumer 工作完全相同。CE consumer 按实验定义
读取预变换 shadow。宿主默认使用每 GPU 一个持久 worker 并行提交，模拟 Flux
一进程一卡；`--serial-host` 仅用于复现/诊断旧的四 rank 串行提交偏差。

```bash
# 快速正确性检查
./exp4_ag_tile_transport --k 1024 --panel-h 1,4,16 --n-comm 4 \
    --iters 5 --verify --modes fused,compute-only,comm-only

# 正式 H sweep：16 KiB -- 2 MiB/flag，总字节数固定
./exp4_ag_tile_transport --k 8192 \
    --panel-h 1,2,4,8,16,32,64,128 --n-comm 8 --verify \
    --modes fused,compute-only,comm-only,bystander,local,memop-cost \
    --warmup-ms 500 --window-ms 200 --csv exp4_panel.csv

# 等 SM 对照：CE 也空出与 n_comm=8 相同的 8 个 compute block
./exp4_ag_tile_transport --k 8192 --panel-h 1,8,128 \
    --variants ce-aggregate,ce-panelized --ce-reserve-sm 8 --verify \
    --modes fused,compute-only,bystander,memop-cost \
    --warmup-ms 500 --window-ms 200 --csv exp4_panel_ce_isosm.csv

# TMA 通信 SM 数校准
./exp4_ag_tile_transport --k 8192 --panel-h 1,8,128 --variants tma \
    --n-comm 1,2,4,8,16 --verify \
    --modes fused,compute-only,comm-only,bystander \
    --warmup-ms 500 --window-ms 200 --csv exp4_panel_ncomm.csv
```

panel 模式下 CSV 仍令 `g_rb=1`，并新增末尾列 `axis=panel` 与 `panel_h=H`；
`chunk_bytes=H×16384` 是实际每 flag 字节数。TMA 使用的通信 block 数自动截断
到实际 job 数，避免大 H 时为空闲 comm block 永久挤掉 compute block。

## 实现备注（对照 flux / CUDA 12.9）

- flux 的 CE 拷贝是 `cudaMemcpyAsync(cudaMemcpyDefault)` 作用在 cudaIPC
  指针上；这里单进程用 `cudaMemcpyPeerAsync`/UVA，走同样的 copy engine 路
  径，不需要 IPC handle 交换。
- flux 的跨卡信号：宿主侧 `cuStreamWriteValue32_v2`（dlopen libcuda）+ 内
  核内 system-scope 自旋。exp1–exp3 的宿主侧编排全部用 CUDA event（无 SM
  开销，CUDA 12.9 无弃用问题），不需要链接 libcuda；exp4 的 CE 旗标是被测
  机制本身，所以照 flux 用 `cuStreamWriteValue32`，但经
  `cudaGetDriverEntryPoint` 运行时解析（CUTLASS 同款机制），仍不链接
  libcuda，拿不到符号时退化为打了 `flag_mech=kernel` 标签的微 kernel。
- 避开了 12.x 已弃用的 `cudaDeviceProp::clockRate`（用
  `cudaDeviceGetAttribute` 查询）；未使用 legacy IPC、NVML NvLink 系列等
  flux 中在新 toolkit 上有摩擦的 API。
- `-arch=sm_90a` 必须带 `a`（WGMMA/TMA），与 flux 的 CMake 处理一致。
- exp2 的拷贝源数据每轮重复，8192 形状下单 shard 32 MiB 可能部分驻留
  L2（H800 50 MiB），t_comm 可能略偏乐观；加大 `--sizes` 的 M/K 可消除。
- 所有实验都支持 `--csv` 追加机器可读结果，便于画图。
