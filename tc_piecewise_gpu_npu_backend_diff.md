# TC Piecewise 第三层：GPU 与 NPU 实现差异

本文对比当前代码中的 `CUDAPiecewiseBackend` 与 `NPUPiecewiseBackend`。NPU 继承 GPU 类的初始化逻辑，主要重写 `__call__()`；这里仅讨论单个 FX 片段的 shape 执行、warmup、capture 和 replay。

| 差异点 | GPU | NPU | 影响 |
| --- | --- | --- | --- |
| **首次调用处理** | 第一次只执行通用 callable，并调用编译结束处理 | 没有专门分支，直接进入 bucket 逻辑 | 首次执行的生命周期不同 |
| **无符号维度保护** | `sym_shape_indices` 为空时直接执行通用 callable | 直接读取 `sym_shape_indices[0]` | NPU 此处要求列表非空 |
| **按 shape 专门编译** | 有 `need_to_compile` 判断和专门编译路径 | 没有对应路径 | 当前默认 `compile_sizes` 为空，通常不体现差异 |
| **编译预热阶段隔离** | 检查 `is_in_torch_compile_warmup()`，此时只执行、不捕获 | 不检查该标记 | NPU 可能在编译预热阶段推进到捕获，具体取决于实际调用次数 |
| **捕获 stream** | 读取公共捕获 stream，并显式传给捕获接口 | 捕获接口没有显式传入 stream | 捕获所用 stream 的控制方式不同，不代表 NPU 不使用 stream |
| **缺少捕获 stream 时回退** | 执行普通 callable，并给出提示 | 没有对应检查 | GPU 对运行时 Dynamo 重编译有明确回退 |
| **设备图接口** | `CUDAGraph`、`torch.cuda.graph` | `NPUGraph`、`torch.npu.graph` | 底层设备图运行时不同 |
| **公共内存池作用域** | 捕获使用 `graph_pool_capture_scope()`，replay 使用 `graph_pool_replay_scope()` | 没有这两层 scope | 公共资源管理接入不同；两者都向捕获接口传入 `pool`，缺少 scope 本身不能证明存在内存问题 |
| **清理缓存 API** | 非首片段暂时屏蔽 `torch.cuda.empty_cache` | 非首片段暂时屏蔽 `torch.npu.empty_cache` | 设备 API 替换，目的相同：避免每个片段捕获时重复清理缓存 |

## 共同机制

- 使用 `concrete_size_entries[T]` 保存各个 token bucket 的状态。
- 未配置的 shape 执行通用 callable。
- 每个 bucket 在捕获前执行一次普通 warmup。
- 缓存设备图和输出，replay 后返回缓存输出。
- 使用弱引用管理输出，debug 模式检查输入 tensor 地址。
- 此层均未调用设备图 `update()` API；输入刷新由上层 Runner 完成。

## 源码位置

- GPU：`python/sglang/srt/compilation/cuda_piecewise_backend.py`，`CUDAPiecewiseBackend.__call__()`。
- NPU：`python/sglang/srt/compilation/npu_piecewise_backend.py`，`NPUPiecewiseBackend.__call__()`。
- 设备后端选择：`python/sglang/srt/compilation/backend.py`，`make_backend()`。

以上为静态代码对比，未进行 GPU/NPU 实机验证。
