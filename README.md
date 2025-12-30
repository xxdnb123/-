# TargetDiff 快速使用说明

一个基于 3D 等变扩散的蛋白口袋定向分子生成与亲和力预测项目（ICLR 2023）。模型输入蛋白口袋（10Å 裁剪）、可选参考配体信息，输出配体 3D 坐标与原子类型轨迹。当前版本加入了我新增的“纯能量视角”改进：对生成轨迹做几何/价态筛查与力场能量（可选 AMBER）评估，去掉了结构相似度计算，专注能量质量。

## 输入与输出
- 主要输入：`configs/training.yml` / `configs/sampling.yml`（训练与采样超参）、预训练权重 `pretrained_models/pretrained_diffusion.pt`、数据根目录 `data/*`（CrossDocked 预处理及拆分）。
- 采样输出（示例目录 `outputs/similar_and_energy`）：
  - `result_<data_id>.pt`：保存单个 pocket 的生成结果（位置、类别、轨迹等）。
  - `energy_overview_data<data_id>.csv`：每个样本的有效率、末帧能量、末帧 AMBER 能量。
  - `plots/*.png` / `plots/*.json`：逐时间步能量曲线及同名元数据。
  - `sample.yml`：本次采样所用配置快照。

## 轨迹检查脚本的目的（新增能量视角改进）
- `xxd_tools/chem_eval.py`：将生成的 (Z, pos) 轨迹转为分子，做几何/价态快速筛查，可选最小化，并输出力场能量；若环境可用 AMBER，再补充 AMBER 势能。这是我新增的“纯能量视角”改进，已去掉结构相似度计算，专注能量质量。
- `scripts/sample_diffusion.py` / `scripts/post_compute_amber.py`：调用上面的能量评估，生成能量曲线和概览 CSV，便于快速发现无效帧、能量异常样本。

## 环境
推荐直接用本目录的 `my_env.yaml` 创建：
```bash
conda env create -f my_env.yaml
conda activate targetdiff
```
该环境包含 PyTorch 1.13 / CUDA 11.8、PyG、RDKit、OpenMM/OpenFF（用于 AMBER 能量）、SciPy、Matplotlib 等依赖。

## 数据下载
使用夸克网盘下载后，在目录下解压得到/data:
```
我用夸克网盘给你分享了「data.zip」，点击链接或复制整段内容，打开「夸克APP」即可获取。
/~ce4639oIk4~:/
链接：https://pan.quark.cn/s/4c0260b820da
提取码：R2kD
```

## 运行示例
- 单卡采样（举例用 GPU0，data_id=0，batch=8）：
```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.sample_diffusion configs/sampling.yml -i 0 --batch_size 8 --result_path outputs/single_gpu
```
- 多卡采样（用 `batch_sample_diffusion.sh` 按任务号分片，参数：配置、输出目录、总卡数、当前卡序号、起始任务索引；下面是 4 卡示例，每条命令放到对应 GPU 上）：
```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/similar_and_energy 4 0 0
CUDA_VISIBLE_DEVICES=3 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/similar_and_energy 4 1 0
CUDA_VISIBLE_DEVICES=4 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/similar_and_energy 4 2 0
CUDA_VISIBLE_DEVICES=5 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/similar_and_energy 4 3 0
```
- 可选：采样后若仅想重新计算能量/AMBER 并出图，可运行：
```bash
python scripts/post_compute_amber.py --result_path outputs/similar_and_energy --data_id 0
```

## 生成结果目录结构（示例 `outputs/similar_and_energy`）
- `result_<data_id>.pt`：原始生成结果包（坐标、类别、轨迹、idx2Z、计时）。
- `energy_overview_data<data_id>.csv`：按样本汇总的有效率与末帧能量。
- `plots/`：`*.png` 为能量曲线图；同名 `*.json` 存曲线数值和元数据。
- `sample.yml`：运行配置备份，便于复现。
