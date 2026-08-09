# comm_comp — 通信/计算干扰实验（4×H800 NVLink, sm90）

三个独立的 CUDA/CUTLASS 微基准，量化通信与计算在 Hopper 上的资源竞争，
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
nvidia-smi -lgc 1980 -i 0,1,2,3      # H800 图形时钟按机器实际上限调整
```

## 一键跑全部实验 — `run_all.sh`

```bash
sudo nvidia-smi -lgc 1980 -i 0,1,2,3   # 先锁频（脚本只记录时钟，不改）
./run_all.sh                            # 全量，约 20-40 分钟
QUICK=1 ./run_all.sh                    # 快速版
```

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
不空闲；若观察到空闲会在行尾标 `[!]`，此时 S_c 只是下界。

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

```
slowdown = t_fused / t_baseline    （baseline = 同配置 stock GEMM）
```

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
- flux 强制 epilogue `StagesD=1`，这里保留 stock 的 stage 数（逐 tile
  `tma_store_wait<0>` 已经起到同样的排空作用）；DMA smem 计入 mainloop
  stage carveout，所以 fused kernel 的 mainloop stage 数可能比 baseline
  少一档（程序会打印两者）。

**多进程**：一进程一卡（`fork`），buffer 用 cudaIPC 交换（匿名共享 mmap
传 handle），flux 式 device barrier-all 对齐各 rank 的每次迭代——与
torchrun 部署形态一致，不需要 MPI/pybind11。四个 rank 对称执行：每张卡
一边算自己的 GEMM 一边被其它三张卡拉取。仅支持 Linux。

```bash
make exp3_gemm_rs_fused
./exp3_gemm_rs_fused --verify              # 先验证 RS 结果（fp32 参考 + 容差）
./exp3_gemm_rs_fused --m 8192 --n 8192 --k 2048 --csv rs.csv
```

读法：`slowdn` 是融合通信的全部代价（2 个 warp 的 SM 驻留 + flag 同步 +
NVLink 拉取 + 等最慢 peer 的尾部）。与 `exp3_epilogue_remote` 的
remote/local 比值对照，可以回答"flux 在 sm90 上选 pull 而不是 epilogue
远端写，赚了还是亏了"。

## 实现备注（对照 flux / CUDA 12.9）

- flux 的 CE 拷贝是 `cudaMemcpyAsync(cudaMemcpyDefault)` 作用在 cudaIPC
  指针上；这里单进程用 `cudaMemcpyPeerAsync`/UVA，走同样的 copy engine 路
  径，不需要 IPC handle 交换。
- flux 的跨卡信号：宿主侧 `cuStreamWriteValue32_v2`（dlopen libcuda）+ 内
  核内 system-scope 自旋。这里宿主侧编排全部用 CUDA event（无 SM 开销，
  CUDA 12.9 无弃用问题），不需要链接 libcuda。
- 避开了 12.x 已弃用的 `cudaDeviceProp::clockRate`（用
  `cudaDeviceGetAttribute` 查询）；未使用 legacy IPC、NVML NvLink 系列等
  flux 中在新 toolkit 上有摩擦的 API。
- `-arch=sm_90a` 必须带 `a`（WGMMA/TMA），与 flux 的 CMake 处理一致。
- exp2 的拷贝源数据每轮重复，8192 形状下单 shard 32 MiB 可能部分驻留
  L2（H800 50 MiB），t_comm 可能略偏乐观；加大 `--sizes` 的 M/K 可消除。
- 三个实验都支持 `--csv` 追加机器可读结果，便于画图。
