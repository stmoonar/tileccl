# signalling：push vs pull 完成信令开销

复现 MoK（Mixture of Kittens）的核心观察：NVL72 多节点微基准里 push-based
dispatch 的 signalling 延迟约 **103 µs**，pull-based 只有约 **18 µs**（5.8×）。
差异不在数据搬运本身，而在**完成判定协议**——push 需要跨 GPU 的"数据就绪"
通知（fence + 远端 flag + 接收端 P-way fan-in 等待）；pull 把数据到达本身
当作完成事件（请求方本地判定，单一实体管理）。

原方案是 NVSHMEM 写法；本实现是它的单进程纯 CUDA 翻译（与本仓库其他工具
同风格，不引入 launcher 和库依赖），原语一一对应：

| NVSHMEM | 本实现 |
|---|---|
| `nvshmemx_signal_op` | `st.release.sys` 写对端 flag |
| `nvshmem_putmem_signal_nbi` | peer `uint4` store 流 + `st.release.sys` flag |
| put + `nvshmem_fence` + signal | peer store + `__threadfence_system` + 普通 store |
| `nvshmem_getmem_nbi` + `quiet` | `__ldcv` 远端 load + 本地完成计数 |
| `nvshmem_uint64_wait_until` | `ld.acquire.sys` 轮询 |

## 角色与逻辑 fan-in

一张 **target** 卡（计时都在它上面），其余卡是 **source**。MoK 的 71-peer
fan-in 在几张卡的机器上测不了，用**逻辑源**模拟：每张 source 卡跑 M 个独立
CTA，各有自己的 payload slot 和自己的 flag，target 等全部 F = ΣM 个。协议
形状（F 个独立 producer、F-way fan-in 等待）保持不变；代价是这些逻辑源共享
物理链路——这正是要测的 fan-in 聚合效应，但解读时记住带宽是共享的。

## 五种 mode（计时窗口内是什么）

| mode | 计时窗口 | 回答的问题 |
|---|---|---|
| `signal` | go → 源发 `st.release.sys` flag → F-way 等待 | 纯 fan-in 下限，无任何 payload |
| `post` | payload **提前送达并确认**后：go → flag → 等待 | "数据已经躺在目的地、还在等 signal"的空窗。协议与 `signal` 完全相同，超出部分 = 刚送完 payload 的残余排队/drain |
| `push` | go → payload store → release-flag → F-way 等待 | push dispatch 完整完成协议 |
| `push_fence` | 同上，但 flag 换成显式 `__threadfence_system` + 普通 store | 融合 release-store vs 手写 fence 序列的差价（与 `sync_cost` 的 d(rel)/d(fence) 互为印证） |
| `pull` | target 的 F 个 CTA 各自 `__ldcv` 读远端 slot → 本地 store → 本地完成计数归零 | pull dispatch：完成判定全部本地化，全程无跨卡 flag |

`pull` 用 `ld.global.cv`（`__ldcv`）绕过本地 L2，保证每轮真的重新过链路取数，
而不是命中上一轮的缓存行。

## 计时与 go 握手

- target 在 kernel 内用 `clock64()` 逐轮计时（从不比较两卡时间戳），启动时用
  自旋 kernel + CUDA event 标定 cycles/µs。**锁频**，否则 DVFS 会让换算漂移。
- 每轮由 target 向每个逻辑源 release 一个 `go` flag 来对齐节奏（否则源会
  自由跑到接收端前面去）。因此 `signal`/`post`/`push`/`push_fence` 的计时
  窗口里含一跳 target→source 单向延迟：
  - 这四个 mode **互相之间的差值是干净的**（加数相同）；
  - 与 `pull`（无握手）比**绝对值**时，从前四者中减去 `sync_cost` ping-pong
    的单向延迟。
- `pull` 的 `t0` 读取后有**第二次 grid barrier**，保证时间戳先于任何 CTA
  的第一条远端 load（否则其它 CTA 会抢跑在 t0 之前，pull 被系统性少计
  ——偏差是 barrier 释放偏斜量级，对小消息点占比可观）。计时窗因此含一次
  barrier release，与 push 侧计时窗内的 `__syncthreads` release 对称。
- 逐轮样本报 min / p50 / p95 / p99。fan-in 成本是尾部现象（等最慢的那个
  peer），均值会骗人。

## 构建与运行

```bash
cd signalling
make                    # 默认 -arch=sm_120a（RTX PRO 5000）
make ARCH=sm_90a        # H20

./signal_fanin                                        # 全 5 mode × 默认扫描
./signal_fanin --modes push,pull --fanin 1,2,4,8,16,32 --sizes 4K,8K,32K,128K
./signal_fanin --modes signal,post --fanin 1,8,32,64
./signal_fanin --csv signalling.csv
```

| 参数 | 说明 |
|---|---|
| `--modes LIST` | `signal,post,push,push_fence,pull` 的子集 |
| `--sizes LIST` | 每个逻辑源的消息大小（16 B 的倍数，K/M/G 后缀）。`signal` 忽略 |
| `--fanin LIST` | 逻辑源总数（≤1024），round-robin 摊到可用 source 卡上 |
| `--sources LIST` / `--target N` | 手工指定角色；默认 target=0，其余能 P2P 的都当 source |
| `--threads N` | 每 CTA 线程数（payload 搬运并行度），默认 256 |
| `--rounds` / `--warmup` | 计入统计的轮数 / 预热轮数，默认 300 / 50 |
| `--no-check` | 跳过 payload 校验 |
| `--csv FILE` | 机器可读输出 |

## 读数方式

按 MoK 的论证逐条对质：

1. **`signal` vs fanin**：纯 fan-in 等待怎么随 P 增长。这是 push 协议的
   下限；如果它已经陡增，push 的 103 µs 主要就是 fan-in 尾部。
2. **`post` − `signal`**：同一协议、唯一区别是刚送完 S 字节 payload。差值
   显著 → 前面的远端写没排干净就开始发 signal（drain 成本被计入 signal）。
3. **`push` 随 size 增长、`signal`/`post` 不随**：验证"signal 不能先于
   payload 可见，所以 push 协议必然背上传输时间"。
4. **`push` − `pull` 同 (S, P)**：方向差 + 协议差的总和。注意先从 push 侧
   减掉 go 握手的单向延迟再比。
5. **`push_fence` − `push`**：显式 fence 序列比融合 release-store 贵多少。
6. **verify 列**：push 系检查 slot 内容 + 末轮 epoch 戳（证明最后一轮的写
   真的落地且序正确）；pull 检查读回内容。`FAIL` 说明该路径的可见性语义
   有问题——这本身就是结论。

预期形状（校验数据合理性）：`signal` 的 p50 ≈ ping-pong RTT、随 P 缓增而
p99 明显放大；`post` ≈ `signal`（若不是，检查 pre-flag 握手）；小 S 时
`push` ≈ `signal` + 单程传输，大 S 时被带宽主导；`pull` 小 S 时 ≈ 单程读
延迟（无 RTT 加数），且随 P 的增长主要来自链路共享而非协议。

## 注意事项

1. **锁频**：`nvidia-smi -lgc`，理由同 `tests/README.md`。cycles/µs 标定值
   会打印出来，跑前后各看一眼是否一致。
2. 所有 kernel 都在自旋等 flag，Windows/WDDM 有 watchdog 风险；在 Linux
   服务器上跑。轮数大、payload 大时单个配置的 kernel 会跑几百 ms。
3. `pull` 用 cooperative launch 保证 F 个 CTA 共驻（软件 grid barrier 的
   前提）；F 超过驻留容量的配置会打 skip。
4. 本机是 PCIe 或单机 NVLink，测不出 NVL72 的绝对数值；能测的是**协议形状**
   ——push 的 fan-in/fence/drain 成本怎么随 P 和 S 增长、pull 为什么不随。
5. 逻辑源共享物理链路：同一张卡上的 M 个 CTA 同时推 payload 会互相挤带宽。
   对 `signal`/`post`（无 payload / payload 不计时）这不污染结论；对
   `push`/`pull` 的大 S 点，fan-in 维度和带宽维度是耦合的，配合
   固定 S·P 总量的扫法（`--sizes` 除以 P）可以解耦。
6. 与 `tests/sync_cost` 的关系：那边测的是单对单的原语成本（ping-pong、
   fence 边际、远端 atomic）；这边把它们组装成完整的完成协议并加上 fan-in
   维度。两边数字应当自洽（如 `signal` P=1 的 p50 ≈ ping-pong RTT）。
