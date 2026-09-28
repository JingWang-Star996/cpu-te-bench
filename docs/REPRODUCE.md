# 运行方法与复现边界

## 先分析已有记录：不会启动模型

Python 3.10+，无需第三方包：

```powershell
python scripts/analyze_recorded.py
python -m unittest discover -s tests -v
```

输出由发布的原始计时记录重新计算，默认不改文件。

## 自己运行编码器

脚本是这次测试所用文件的原样副本，定位为Windows/ComfyUI实验工具，不是跨平台安装器或生产服务。它要求已有且可用的ComfyUI Python环境，包含PyTorch、NumPy、Pillow、psutil及相关编码器/量化依赖。请使用ComfyUI自身的Python，不要为运行本仓库盲目更换正在工作的PyTorch。

已测PyTorch为2.13.0+cu130，Python 3.13.14；详细软件与六个源码SHA见runs.json。H3 GPU使用传统ModelPatcher，显式禁用DynamicVRAM，不代表完整ComfyUI main启动后的所有路径。CPU使用--cpu与FP32配置，禁止CUDA初始化。

以下路径是使用者自行设置的示例，脚本不修改ComfyUI源码、不启动HTTP服务、不运行DiT或VAE。先确保设备空闲；脚本不会替你排空生产队列、卸载服务或抢占任务。

```powershell
$python = '.\ComfyUI_windows_portable\python_embeded\python.exe'
$comfy = (Resolve-Path '.\ComfyUI_windows_portable\ComfyUI').Path
$imageWeights = Join-Path $comfy 'models\text_encoders\qwen3vl_8b_int8_convrot.safetensors'
$h3Weights = Join-Path $comfy 'models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'

# 每次使用新的输出目录。Image的reference参数是历史接口要求，该用例不会读取它。
& $python -s .\scripts\bench.py --root $comfy --checkpoint $imageWeights --profile image --device cpu --threads 24 --reference unused --output .\runs\image-cpu
& $python -s .\scripts\bench.py --root $comfy --checkpoint $imageWeights --profile image --device cuda --threads 24 --reference unused --output .\runs\image-gpu

# 用你有权使用的参考图。不同图片不能与本仓库历史H3成绩作严格同输入比较。
& $python -s .\scripts\bench.py --root $comfy --checkpoint $h3Weights --profile h3 --device cpu --threads 24 --reference .\reference.png --output .\runs\h3-cpu
& $python -s .\scripts\bench.py --root $comfy --checkpoint $h3Weights --profile h3 --device cuda --threads 24 --reference .\reference.png --output .\runs\h3-gpu --diagnose
& $python -s .\scripts\bench.py --root $comfy --checkpoint $h3Weights --profile h3 --device cuda --threads 24 --reference .\reference.png --output .\runs\h3-gpu-reserve3 --reserve-vram 3 --diagnose
```

这些命令分别启动真实编码，应逐条运行。--diagnose会在4次正式编码后多执行1次Profiler前向；这一次不计入成绩。线程数是实验变量，请按整机能力选择；历史12700H使用14，另外两台使用24。

脚本对两份指定权重校验完整SHA，不自动下载权重，其他权重会被拒绝。H3还校验[1,441,5120]及441个标签；不同图片/预处理若导致形状不符将保存失败并退出。完整文件SHA会预热系统文件缓存，首次forward不是真正磁盘冷启动。

## 输入与来源

Image短文本与长文本（固定句子重复53次）、H3提示词均在bench.py中。H3参考图统一处理为864×480；原始图片、条件张量、模型权重不随本仓库分发。参考图及处理后张量的SHA、输出张量SHA保留在runs.json。

因此：Image文本用例和分析可以直接复用；H3方法可以在自有图片上重复执行，但外部使用者没有原图时不能逐字节重现我们的历史H3样本。发布的CPU/GPU差异是当时保存张量的测量值，仓库无法脱离这些未分发张量重新计算全部差异。

## 内存与资源

本轮整机内存为40/48/48GiB DDR5，非统一64GiB。运行前留足系统余量：内存每0.5秒采样，若连续约2秒可用RAM低于3GiB或运行超过2100秒，独立脚本会退出并留下状态。这个保护不能保证避免全部OOM或换页。

CPU-only计算不需要CUDA算子，但历史脚本仍尝试用nvidia-smi监控GPU；没有该命令时会记录监控错误。这份原样脚本只在配有NVIDIA显卡的Windows电脑上验证过“CPU计算且CUDA不初始化”，尚未在物理无GPU主机验证安装流程。

新增运行结果含本机路径/主机名；分享前请自行脱敏，不要直接提交runs目录。仓库不附带远程控制、服务启停或防火墙脚本。
