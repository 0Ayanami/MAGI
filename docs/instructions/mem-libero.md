# 双记忆原型的 LIBERO 复现

工作根目录为 `/home2/wanquan/VoLoAgent`。在 `codex/mem-voloagent` 的同一提交运行所有组。模型、回放和缓存位于 `data/mem`，结果位于 `results/mem`。两个目录均由 Git 忽略。

## 离线配置与检查

在根目录执行下列命令。LIBERO 使用 Python 3.11 和 CPU PyTorch。代理与仿真客户端共用此环境。OpenPI 使用自己的锁定环境。下载软件包和资产需要网络，预检只读取本地文件。

```bash
stage="$PWD/data/mem"
mkdir -p "$stage"/{tmp,cache,src,config,datasets,logs}
export TMPDIR="$stage/tmp" UV_CACHE_DIR="$stage/cache/uv" XDG_CACHE_HOME="$stage/cache"
export LIBERO_CONFIG_PATH="$stage/config"
librev=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
openrev=215abfb217dbac7d5f1273282331b9b1866c0479
curl -fLsS "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/$librev" -o "$stage/tmp/libero.tgz"
echo "0b0daa3659070f929d6032534d193b36b68acd7239b5a33bfc5472ba621ec18e  $stage/tmp/libero.tgz" | sha256sum -c
tar xzf "$stage/tmp/libero.tgz" -C "$stage/src"
lib="$stage/src/LIBERO-$librev"
base="$lib/libero/libero"
printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\nassets: %s/assets\ndatasets: %s/datasets\n' "$base" "$base" "$base" "$base" "$stage" > "$LIBERO_CONFIG_PATH/config.yaml"
git init "$stage/src/openpi"
git -C "$stage/src/openpi" fetch --depth 1 git@github.com:Physical-Intelligence/openpi.git "$openrev"
git -C "$stage/src/openpi" checkout --detach FETCH_HEAD
export UV_PROJECT_ENVIRONMENT="$stage/openpi-env" GIT_LFS_SKIP_SMUDGE=1
uv sync --directory "$stage/src/openpi" --python 3.11 --frozen --no-dev
uv venv --python 3.11 "$stage/libero-env"
py="$stage/libero-env/bin/python"
uv pip install --python "$py" \
  'torch @ https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl' \
  'torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.20.1%2Bcpu-cp311-cp311-linux_x86_64.whl' \
  'numpy<2' h5py==3.16.0 matplotlib==3.10.7 robosuite==1.4.1 mujoco==3.3.7 bddl easydict gym==0.26.2 'imageio[ffmpeg]' tyro \
  transformers==4.49.0 pytest google-crc32c pyarrow==18.1.0 -e . -e "$lib" -e "$stage/src/openpi/packages/openpi-client"
"$py" scripts/mem_checkpoint.py
mkdir -p "$stage/cache/openpi/big_vision" "$stage/gdino"
curl -fLsS https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o "$stage/cache/openpi/big_vision/paligemma_tokenizer.model"
rev=a2bb814dd30d776dcf7e30523b00659f4f141c71
for f in model.safetensors config.json preprocessor_config.json added_tokens.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
  curl -fLsS --retry 2 "https://hf-mirror.com/IDEA-Research/grounding-dino-tiny/resolve/$rev/$f" -o "$stage/gdino/$f"
done
echo "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3  $stage/gdino/model.safetensors" | sha256sum -c
bash scripts/mem_libero.sh preflight
"$py" -m pytest -q tests/test_episode_memory.py tests/test_subgoal.py tests/test_libero.py tests/test_libero_benchmarks.py
```

## 既有示范与受控重规划

使用已发布数据集 `HuggingFaceVLA/libero` 的固定 revision，取双杯任务 episode 0。这段示范参与了白杯颜色筛选和相机镜像方向的核验，应作为开发集报告。视觉回放使用真实 RGB、机器人状态和固定外部相机标定，模拟云监控返回 continue。语义回放向真实云模型发送相同的两幅图，分别注入空摘要、曾经放稳、放稳后又失效三种条件。

```bash
mkdir -p data/mem/replay results/mem/recorded
curl -fLsS https://hf-mirror.com/datasets/HuggingFaceVLA/libero/resolve/86958911c0f959db2bbbdb107eb3e17c5f9c798e/data/chunk-000/file-000.parquet -o data/mem/replay/file-000.parquet
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 bash scripts/mem_libero.sh recorded > results/mem/recorded/vision.log
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 bash scripts/mem_libero.sh recorded visual enforce > results/mem/recorded/vision-enforce.log
```

本地运行 `.venv\Scripts\python.exe scripts/mem_cloud.py replay baseline controlled` 获取三次已标记的真实云端重规划。该命令使用同一个原版 recycle 提示词和输出解析器，返回剩余技能列表。

## 闭环运行命令

获得实验指令后，在本地运行一次图像探针。启动器从本地 `.env` 读取 `DASHSCOPE_API_KEY`，经 SSH stdin 注入服务器进程。北京是配置默认值，新加坡凭据使用 `--region singapore`。探针会实际调用快照模型 `qwen3.5-plus-2026-04-20`。

```powershell
.venv\Scripts\python.exe scripts/mem_cloud.py probe baseline probe
```

每组单独启动策略服务，令 π0.5 的采样状态从同一起点开始。在服务器选用当时可用的 GPU，五组使用相同的设备设置。按下面顺序在三个终端运行。每组完成后结束该组启动的代理和策略服务。重复运行使用新的批次名。

```bash
# 服务器终端 1
cd /home2/wanquan/VoLoAgent
bash scripts/mem_libero.sh policy
```

```powershell
# 本地终端 2
.venv\Scripts\python.exe scripts/mem_cloud.py proxy baseline pair01
```

```bash
# 服务器终端 3
bash scripts/mem_libero.sh eval baseline pair01
data/mem/libero-env/bin/python scripts/mem_score.py results/mem/pair01/baseline
```

| 组及结果目录（相对批次目录） | 语义 | 视觉 | 输入 |
|---|---|---|---|
| baseline | off | off | history |
| semantic | episode | off | history |
| visual | off | enforce | history |
| dual | episode | enforce | history |
| current | off | enforce | current |

单杯诊断和双杯任务均使用 seed 7、初态 0、原套件终止条件和 5 步动作块，基础检查间隔 40 步。端侧检测使用固定相机 RGB 和关节状态。每次检查输出的 `placement_check` 记录包含源帧步号。影子模式可通过原 CLI 指定 `--visual-memory shadow`。诊断阈值在离线回放期设定。

评分数据只在客户端记录。放稳的操作定义为连续三个仿真步满足对应 On 谓词、脱离抓握、物体相对盘的位置变化在 5 mm 内。评分以带下一对象指令的实际执行动作为交接，输出分子和分母。没有可识别交接时比率为 JSON null。另保存所有派发以人工复核未知指令。终止前未确认的放稳事件计入漏确认；确认延迟以仿真步计。累计耗时包含云请求等待；成功与失败回合分别列出。自然 replan 和受控的离线回放分开报告。
