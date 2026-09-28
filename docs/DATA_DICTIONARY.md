# 数据字段

results/2026-09-28/runs.json以CPU型号、profile、device和可选reserve3命名。原始计时数值与本地result.json逐项相等。

- samples：每次完整编码，repeat=1..4；Image有短/长两用例，H3一个固定首帧用例。
- encode_s：分词＋同步完成的编码前向；不含导入、文件SHA、load_clip API、参考图预处理、保存npy与输出数值校验。返回CPU条件张量的内部复制已在forward内。
- tokenize_s / forward_s / output_copy_s：分段耗时，单位秒。
- cuda.peak_allocated_bytes / peak_reserved_bytes：每轮重置后PyTorch分配器峰值，不等于驱动物理显存占用，不可相加。
- memory：0.5秒采样的工作集RSS、私有提交和系统最低可用RAM。overall包括载入与额外Profiler（若启用）；短峰值可能漏采。RSS和私有提交不可相加。
- placement_after：最后一次正式计时后的模块位置与权重装载量；独立Profiler可能继续改变卸载状态。
- checkpoint：文件字节数、完整SHA256；权重校验读取整个文件，会预热文件缓存。
- reference：原图和预处理张量SHA、预处理时间；原图不分发。
- sha256 / token_tags：每次输出张量与H3标签的哈希；原输出npy不分发。
- diagnostic：计时之外的Profiler聚合表；嵌套算子时间可能重复，不应简单求和。
- comparisons.json：当时使用首轮CPU/GPU张量计算的最大绝对差、RMSE、相对RMSE和余弦，以及预留组GPU输出一致性。未设置画质通过阈值。

运行环境原本带有本地目录、主机名、PID、SSH和服务信息，发布版通过字段白名单剔除。该操作没有改变样本数值。未发布原始服务回执、全量遥测和私有输入。
