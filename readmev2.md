# TRELLIS 远程服务器部署与推理详细操作文档

本文档记录 TRELLIS 在远程 Linux 服务器上的完整部署、依赖安装、常见报错修复和推理运行流程。内容面向初学者，命令尽量按实际执行顺序排列。

本文示例环境：

```bash
系统：Linux
GPU：NVIDIA GPU
Python：3.10
Conda 环境名：3D
项目路径：/data1/zhkang/TRELLIS-main
PyTorch：torch==2.4.0
TorchVision：torchvision==0.19.0
TorchAudio：torchaudio==2.4.0
PyTorch CUDA：12.1
```

如果你的用户名、环境名或项目路径不同，请把命令中的 `/data1/zhkang/TRELLIS-main`、`3D` 等替换成你自己的路径和环境名。

## 1. 下载 TRELLIS 项目

推荐直接在服务器上重新 clone 项目，并且必须带 `--recurse-submodules`。TRELLIS 中的 `flexicubes` 是 Git submodule，如果没有正确下载，`trellis/representations/mesh/flexicubes/` 文件夹会是空的，后续 mesh 相关推理会报错。

```bash
cd /data1/zhkang

git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git TRELLIS-main
cd /data1/zhkang/TRELLIS-main
```

如果你已经下载过项目，但发现 `trellis/representations/mesh/flexicubes/` 是空文件夹，执行：

```bash
cd /data1/zhkang/TRELLIS-main

git submodule sync --recursive
git submodule update --init --recursive
```

验证 `flexicubes` 是否存在：

```bash
ls trellis/representations/mesh/flexicubes
```

正常情况下应能看到类似文件：

```text
flexicubes.py
LICENSE.txt
README.md
```

如果服务器无法访问 GitHub，可以在有网络的机器上下载 `https://github.com/MaxtirError/FlexiCubes.git`，然后把整个文件夹上传到：

```text
/data1/zhkang/TRELLIS-main/trellis/representations/mesh/flexicubes/
```

## 2. 创建 Conda 环境

创建 Python 3.10 环境：

```bash
conda create -n 3D python=3.10 -y
conda activate 3D
```

安装 PyTorch。本文使用 CUDA 12.1 版本：

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

验证 PyTorch 是否能识别 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

正常结果类似：

```text
2.4.0+cu121 12.1 True
```

如果最后是 `False`，说明当前环境没有正确识别 GPU，需要先检查 NVIDIA 驱动、CUDA runtime 或 PyTorch 安装版本。

## 3. 修正 CUDA 环境变量

很多服务器的 `~/.bashrc` 中可能写了类似内容：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```

这里第三行是错误写法：

```bash
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```

`CUDA_HOME` 不能像 `PATH` 一样写多个目录，它必须是一个单独目录。错误写法会导致 `CUDA_HOME` 变成：

```text
/data1/zhkang/miniconda3/envs/3D:/usr/local/cuda:/usr/local/cuda
```

这会让 CUDA 扩展编译失败。

建议打开 `~/.bashrc`：

```bash
vim ~/.bashrc
```

把这三行注释掉：

```bash
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export PATH=$PATH:/usr/local/cuda/bin
# export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```

保存后重新加载：

```bash
source ~/.bashrc
conda activate 3D
```

## 4. 安装 CUDA 12.1 编译器

TRELLIS 的若干 CUDA 扩展需要本地编译。本文环境是 `torch 2.4.0+cu121`，因此推荐使用 CUDA 12.1 的 `nvcc` 编译。

先检查当前 `nvcc`：

```bash
which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果 `torch.version.cuda` 是 `12.1`，但 `nvcc --version` 显示 CUDA 13.0 或其他版本，就不建议直接编译扩展。

在当前 conda 环境中安装 CUDA 12.1 toolkit：

```bash
conda activate 3D
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
```

设置当前终端使用 conda 环境中的 CUDA：

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
hash -r
```

验证：

```bash
echo $CUDA_HOME
which nvcc
nvcc --version
ls -l $CONDA_PREFIX/bin/nvcc
```

正常结果应类似：

```text
/data1/zhkang/miniconda3/envs/3D
/data1/zhkang/miniconda3/envs/3D/bin/nvcc
Cuda compilation tools, release 12.1
```

## 5. 安装 TRELLIS 基础依赖

进入项目目录：

```bash
cd /data1/zhkang/TRELLIS-main
conda activate 3D
```

安装基础依赖：

```bash
. ./setup.sh --basic
```

这一步会安装 `pillow`、`imageio`、`easydict`、`opencv-python-headless`、`rembg`、`trimesh`、`open3d`、`xatlas`、`pyvista`、`pymeshfix`、`igraph`、`transformers`、`utils3d` 等。

如果未执行 `--basic`，可能会连续出现：

```text
ModuleNotFoundError: No module named 'easydict'
ModuleNotFoundError: No module named 'rembg'
```

不要一个一个手动补包，直接运行：

```bash
. ./setup.sh --basic
```

## 6. 固定 transformers 版本

如果安装到了过新的 `transformers`，可能和 `torch==2.4.0` 不兼容，出现类似：

```text
ModuleNotFoundError: No module named 'torch.distributed.tensor.device_mesh'
ModuleNotFoundError: Could not import module 'CLIPTextModel'
```

解决方法是降低并固定 `transformers` 和 `tokenizers` 版本：

```bash
python -m pip uninstall -y transformers tokenizers
python -m pip install "transformers==4.44.2" "tokenizers==0.19.1" "huggingface_hub>=0.23.2,<1.0"
```

验证：

```bash
python - <<'PY'
import torch
import transformers
from transformers import CLIPTextModel, AutoTokenizer

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("CLIPTextModel OK")
PY
```

正常输出应包含：

```text
transformers: 4.44.2
CLIPTextModel OK
```

## 7. 安装 xformers 注意力后端

TRELLIS 默认可用 `flash-attn`，但在很多服务器上 `flash-attn` 编译麻烦。本文推荐先使用 `xformers`，部署更稳。

本文环境是 `torch 2.4.0+cu121`，安装：

```bash
python -m pip install "xformers==0.0.27.post2" --index-url https://download.pytorch.org/whl/cu121 --no-deps
```

验证：

```bash
python - <<'PY'
import torch
import xformers
import xformers.ops as xops

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("xformers:", xformers.__version__)
print("xformers OK")
PY
```

后续运行推理时设置：

```bash
export ATTN_BACKEND=xformers
```

如果不设置并且没有安装 `flash-attn`，可能会出现：

```text
ModuleNotFoundError: No module named 'flash_attn'
```

或：

```text
ModuleNotFoundError: No module named 'xformers'
```

## 8. 安装 spconv

安装 sparse convolution 依赖：

```bash
cd /data1/zhkang/TRELLIS-main
conda activate 3D

. ./setup.sh --spconv
```

如果你的 PyTorch CUDA 是 12.x，脚本通常会安装 `spconv-cu120`。运行时建议设置：

```bash
export SPCONV_ALGO=native
```

运行中如果看到类似：

```text
FutureWarning: torch.cuda.amp.custom_fwd is deprecated
FutureWarning: torch.cuda.amp.custom_bwd is deprecated
```

这些是 `spconv` 内部 API 的提醒，可以忽略，不影响推理。

## 9. 安装 kaolin

`flexicubes` 需要 `kaolin`。如果报错：

```text
ModuleNotFoundError: No module named 'kaolin'
```

执行：

```bash
python -m pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
```

验证：

```bash
python - <<'PY'
import torch
import kaolin
from kaolin.utils.testing import check_tensor
from trellis.representations.mesh.flexicubes.flexicubes import FlexiCubes

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("kaolin OK")
print("FlexiCubes OK")
PY
```

## 10. 安装 GCC 12 编译器

CUDA 12.1 不支持太新的 GCC。如果你当前系统 GCC 是 13.x，编译 CUDA 扩展时可能会出现：

```text
unsupported GNU version! gcc versions later than 12 are not supported
```

如果有 sudo 权限，可以安装系统 GCC 12：

```bash
sudo apt update
sudo apt install -y gcc-12 g++-12
```

如果没有 sudo 权限，用 conda 安装 GCC 12：

```bash
conda activate 3D
conda install -y -c conda-forge "gcc_linux-64=12.*" "gxx_linux-64=12.*"
```

设置编译器：

```bash
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
```

验证：

```bash
$CC --version
$CXX --version
echo $NVCC_PREPEND_FLAGS
```

正常情况下应显示 GCC/G++ 12.x，并且：

```text
-ccbin /data1/zhkang/miniconda3/envs/3D/bin/x86_64-conda-linux-gnu-g++
```

## 11. 安装渲染相关 CUDA 扩展

TRELLIS 生成视频、导出 GLB、渲染 Gaussian/Radiance Field/Mesh 时，需要以下扩展：

- `nvdiffrast`
- `diffoctreerast`
- `diff_gaussian_rasterization`

先准备源码目录：

```bash
mkdir -p /tmp/extensions
```

克隆扩展：

```bash
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
```

如果提示目录已存在，例如：

```text
fatal: destination path '/tmp/extensions/nvdiffrast' already exists and is not an empty directory.
```

可以忽略，说明之前已经下载过源码。

设置编译环境：

```bash
cd /data1/zhkang/TRELLIS-main
conda activate 3D

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export MAX_JOBS=4
```

清理旧 build：

```bash
rm -rf /tmp/extensions/nvdiffrast/build
rm -rf /tmp/extensions/diffoctreerast/build
rm -rf /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/build
```

安装扩展。注意：这里必须加 `--no-build-isolation`，否则临时构建环境里看不到当前 conda 环境的 `torch`，会报 `ModuleNotFoundError: No module named 'torch'`。

```bash
python -m pip install -v /tmp/extensions/nvdiffrast --no-build-isolation
python -m pip install -v /tmp/extensions/diffoctreerast --no-build-isolation
python -m pip install -v /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization --no-build-isolation
```

验证：

```bash
python - <<'PY'
import nvdiffrast.torch as dr
import diffoctreerast
import diff_gaussian_rasterization

print("render extensions OK")
PY
```

如果输出：

```text
render extensions OK
```

说明渲染扩展安装成功。

## 12. 修复 GitHub API 未登录触发限流

TRELLIS 图像到 3D 的 pipeline 会通过 `torch.hub.load` 加载 DINOv2 图像编码器：

```python
torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
```

如果服务器访问 GitHub 触发匿名 API 限流，可能报：

```text
urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
```

最快修法是让 `torch.hub.load` 跳过 GitHub fork 校验。

编辑文件：

```bash
vim /data1/zhkang/TRELLIS-main/trellis/pipelines/trellis_image_to_3d.py
```

找到 `_init_image_cond_model` 函数中的这一行：

```python
dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
```

改成：

```python
dinov2_model = torch.hub.load(
    'facebookresearch/dinov2',
    name,
    pretrained=True,
    skip_validation=True,
)
```

保存后重新运行推理。

另一种方式是设置 GitHub token：

```bash
export GITHUB_TOKEN=你的github_token
```

但对部署来说，直接加 `skip_validation=True` 通常更省心。

## 13. 运行 TRELLIS 图像到 3D 推理

每次运行前建议设置以下环境变量：

```bash
cd /data1/zhkang/TRELLIS-main
conda activate 3D

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
```

运行官方示例：

```bash
python example.py
```

首次运行会下载 TRELLIS 权重和 DINOv2 权重，可能看到类似：

```text
pipeline.json
ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors
ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors
dinov2_vitl14_reg4_pretrain.pth
Sampling: 100%
Rendering: ...
```

如果命令正常结束，当前目录会生成：

```text
sample_gs.mp4
sample_rf.mp4
sample_mesh.mp4
sample.glb
sample.ply
```

## 14. 输出文件含义

`sample_gs.mp4`

3D Gaussian 表示的旋转预览视频。主要用于查看颜色、外观和整体视觉效果。

`sample_rf.mp4`

Radiance Field 表示的旋转预览视频。类似神经辐射场/体渲染效果，用于查看体积外观和细节。

`sample_mesh.mp4`

Mesh 网格表示的旋转预览视频，通常显示法线或几何结构。主要用于查看模型形状是否合理。

`sample.glb`

通用 3D 模型文件，推荐实际使用。可以导入 Blender、Unity、Unreal、网页 glTF 查看器等。

`sample.ply`

3D Gaussian/点云类文件，更偏研究和中间结果。可以用 MeshLab、CloudCompare、Blender 查看，但普通软件不一定能正确显示 Gaussian 渲染效果。

简单理解：

```text
mp4 = 给人看的预览视频
glb = 可实际使用和导入软件的 3D 模型
ply = 高斯/点云类结果，偏中间表示或研究用途
```

## 15. 查看生成结果

查看视频：

```bash
ls -lh sample*.mp4
```

如果本地电脑查看，可以从服务器下载：

```bash
scp zhkang@服务器IP:/data1/zhkang/TRELLIS-main/sample_gs.mp4 .
scp zhkang@服务器IP:/data1/zhkang/TRELLIS-main/sample_rf.mp4 .
scp zhkang@服务器IP:/data1/zhkang/TRELLIS-main/sample_mesh.mp4 .
scp zhkang@服务器IP:/data1/zhkang/TRELLIS-main/sample.glb .
scp zhkang@服务器IP:/data1/zhkang/TRELLIS-main/sample.ply .
```

查看 `sample.glb`：

- Windows 自带 3D 查看器
- Blender
- Unity
- Unreal Engine
- 在线查看器：https://gltf-viewer.donmccurdy.com/
- VS Code 插件：glTF Tools

查看 `sample.ply`：

- MeshLab
- CloudCompare
- Blender 导入 PLY

实际使用时优先使用：

```text
sample.glb
```

## 16. 部署 Gradio Web Demo

安装 demo 依赖：

```bash
cd /data1/zhkang/TRELLIS-main
conda activate 3D

. ./setup.sh --demo
```

编辑 `app.py`：

```bash
vim /data1/zhkang/TRELLIS-main/app.py
```

找到最后：

```python
demo.launch()
```

改为：

```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

运行：

```bash
python app.py
```

浏览器访问：

```text
http://服务器IP:7860
```

如果服务器端口不能直接开放，可以用 SSH 隧道：

```bash
ssh -L 7860:127.0.0.1:7860 用户名@服务器IP
```

然后在本地浏览器打开：

```text
http://127.0.0.1:7860
```

## 17. 常见问题汇总

### 17.1 flexicubes 文件夹为空

现象：

```text
trellis/representations/mesh/flexicubes/
```

文件夹为空。

解决：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

如果服务器不能访问 GitHub，手动下载 `FlexiCubes` 并上传到对应目录。

### 17.2 No module named easydict/rembg

说明基础依赖没装完整。

解决：

```bash
. ./setup.sh --basic
```

### 17.3 transformers 导入 CLIPTextModel 失败

现象：

```text
ModuleNotFoundError: Could not import module 'CLIPTextModel'
```

解决：

```bash
python -m pip uninstall -y transformers tokenizers
python -m pip install "transformers==4.44.2" "tokenizers==0.19.1" "huggingface_hub>=0.23.2,<1.0"
```

### 17.4 No module named kaolin

解决：

```bash
python -m pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
```

### 17.5 No module named xformers

解决：

```bash
python -m pip install "xformers==0.0.27.post2" --index-url https://download.pytorch.org/whl/cu121 --no-deps
export ATTN_BACKEND=xformers
```

### 17.6 flash-attn 安装失败

如果不是必须使用 `flash-attn`，可以直接使用 `xformers`：

```bash
export ATTN_BACKEND=xformers
```

如果一定要安装 `flash-attn`，需要：

```bash
python -m pip install flash-attn --no-build-isolation
```

但实际部署中建议优先跑通 `xformers`。

### 17.7 CUDA 扩展构建时报 No module named torch

现象：

```text
ModuleNotFoundError: No module named 'torch'
```

原因是 pip 构建隔离环境看不到当前 conda 环境中的 torch。

解决：本地扩展安装时加：

```bash
--no-build-isolation
```

例如：

```bash
python -m pip install -v /tmp/extensions/nvdiffrast --no-build-isolation
```

### 17.8 nvcc 路径错误

现象：

```text
No such file or directory: '/data1/zhkang/miniconda3/envs/3D/bin/nvcc'
```

解决：

```bash
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
hash -r

which nvcc
nvcc --version
```

确保 `nvcc` 是 conda 环境中的 CUDA 12.1。

### 17.9 unsupported GNU version

现象：

```text
unsupported GNU version! gcc versions later than 12 are not supported
```

解决：

```bash
conda install -y -c conda-forge "gcc_linux-64=12.*" "gxx_linux-64=12.*"

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
```

### 17.10 GitHub 403 rate limit exceeded

现象：

```text
urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
```

解决：编辑：

```bash
vim /data1/zhkang/TRELLIS-main/trellis/pipelines/trellis_image_to_3d.py
```

把：

```python
dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
```

改为：

```python
dinov2_model = torch.hub.load(
    'facebookresearch/dinov2',
    name,
    pretrained=True,
    skip_validation=True,
)
```

### 17.11 spconv FutureWarning

现象：

```text
FutureWarning: torch.cuda.amp.custom_fwd is deprecated
FutureWarning: torch.cuda.amp.custom_bwd is deprecated
```

这是 `spconv` 的内部 API 提醒，不影响推理，可以忽略。

## 18. 一键运行前环境变量模板

每次新开终端后，可先执行：

```bash
cd /data1/zhkang/TRELLIS-main
conda activate 3D

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export MAX_JOBS=4
```

然后运行：

```bash
python example.py
```

## 19. 最终成功标志

如果命令能跑完，并且当前目录出现以下文件：

```text
sample_gs.mp4
sample_rf.mp4
sample_mesh.mp4
sample.glb
sample.ply
```

说明 TRELLIS 图像到 3D 推理部署成功。
