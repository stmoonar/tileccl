# 4×H20 基准测试交付包

两套微基准在 4×H20 (NVLink, sm90, 锁频 1830 MHz, CUDA 12.9) 上的运行结果。

## 内容

| 路径 | 说明 |
|---|---|
| `comm_comp_RESULTS_4xH20.md` | comm_comp 四个实验的汇总文档 |
| `comm_comp_results/` | comm_comp 原始产物：日志/CSV/env/打包（run_all.sh 生成） |
| `signalling_RESULTS_4xH20.md` | signalling push-vs-pull 实验汇总文档 |
| `signalling_results/` | signalling 原始产物：signal_fanin.csv + .log |

## 环境

- 4× NVIDIA H20，sm_90，78 SM，NVLink NV18 全互联
- CUDA 12.9 (nvcc V12.9.41)，-arch=sm_90a
- `nvidia-smi -lgc 1980 -i 0,1,2,3` 锁频，运行中 SM clock 稳定 1830 MHz
- git commit 06cb8b0 (bench/p2p-ce-vs-tma)

## 一句话结论

- comm_comp：CE 流量与独立 GEMM 互不干扰；AG+GEMM 取最粗粒度最优；flux sm90 pull 式 RS 相比 sm80 epilogue 远端写在低 K 下快 1.73×。
- signalling：协议形状复现 MoK（push 背传输+fan-in 成本、pull 带宽主导），但 4 卡/32 路逻辑 fan-in 测不出 NVL72 的 5.8× 绝对倍数。
