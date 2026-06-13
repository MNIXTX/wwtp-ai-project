# WWTP AI 智慧水务管控平台

污水处理厂智能优化控制系统 — ASM1 机理模型 + TFT 时序预测 + LightGBM 基线 + PPO 强化学习

---

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10/11 x64 | Windows 11 x64 |
| Python | 离线: 3.10 / 3.14<br>在线: 3.10 ~ 3.14 全系列 | 3.10（离线部署）/ 3.14（开发） |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 5 GB（含离线包） | 10 GB |
| 网络 | 可完全离线运行 | 在线（自动更新依赖） |

---

## 快速开始（新电脑）

### 方法一：一键安装（推荐）

```batch
双击 install.bat
```

脚本自动完成：
1. 检测网络 → 在线/离线模式
2. 查找 Python 3.10+（遍历已知路径 → PATH 检测 → 离线安装器兜底）
3. 创建虚拟环境 `venv\`（`--copies` 模式，避免权限问题）
4. 安装全部依赖（离线模式自动检查 Python 版本是否匹配离线包）
5. 验证安装（11 个核心模块 + PyTorch + 可选模块）

| 模式 | Python 版本 | 说明 |
|------|------------|------|
| 离线 | 3.10、3.14 | 从 `offline_packages\` 安装，无需网络 |
| 离线 | 3.11/3.12/3.13 | 提示不支持，引导安装 Python 3.10 或联网 |
| 在线 | 3.10 ~ 3.14 | 从 PyPI 下载，离线包版本不匹配时自动切换 |

### 方法二：手动安装（在线）

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# PyTorch CPU 版
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 启动 / 停止

| 操作 | 方式 | 说明 |
|------|------|------|
| **启动** | 双击 `start.bat` | 打开控制台窗口显示服务器日志，浏览器自动打开 |
| **停止** | 双击 `stop.bat` | 终止后台进程 |

启动后浏览器访问 `http://127.0.0.1:8501`

### 手动启动（查看详细错误）

```bash
venv\Scripts\python.exe -m streamlit run ui/app.py --server.port=8501
```

### 系统诊断

```bash
venv\Scripts\python.exe scripts\diagnose.py
```

检查 9 大类 23 项：Python 环境、依赖、配置、数据、模型文件完整性、归一化器、磁盘、端口、启动测试。

---

## 界面功能

| 页面 | 功能 |
|------|------|
| **数据浏览与看板** | SCADA 数据浏览、趋势图表、AI 预测 |
| **模型训练与校准** | 交互式数据源选择、一键训练 LightGBM / TFT / PPO、实时日志追踪 |
| **系统配置管理** | 在线 YAML 编辑器、热重载、系统清理 |
| **手动预测** | 输入工艺参数，双模型交叉验证预测出水 COD |

界面支持 **中 / 英 双语切换**，侧边栏底部切换语言。

---

## 文件夹结构

```
WWTP_AI_System/
│
├── install.bat              # 一键安装（在线+离线）
├── start.bat                # 启动 Streamlit 服务
├── stop.bat                 # 停止服务
├── launcher.py              # 启动器入口
├── config.yaml              # 全局配置文件
├── requirements.txt         # Python 依赖清单
│
├── asm1_ode_solver.py       # ASM1 活性污泥 ODE 求解器 (numba JIT)
├── asm1_ppo_env.py          # PPO 强化学习环境（曝气/回流控制）
├── asm1_calibration.py      # ASM1 动力学参数自动校准（差分进化）
├── tft_pure_pytorch.py      # TFT 时序预测模型（PyTorch + ONNX 导出）
├── lgbm_baseline.py         # LightGBM 基线 + 双模型共识验证
├── lgbm_feature_builder.py  # 特征工程（滞后/滚动/时间特征）
│
├── data_pipeline.py         # 数据管道（清洗/对齐/插值/序列构建）
├── inference.py             # 融合推理引擎
├── predictor_gateway.py     # 预测网关（ONNX + LightGBM 双引擎，文件完整性预检）
├── predictor_adapter.py     # 预测适配器（UI 桥接）
├── config_manager.py        # 配置管理（Pydantic 校验 + 热重载）
│
├── train.py                 # PPO 策略训练
├── train_tft.py             # TFT 模型训练
├── run_lgbm.py              # LightGBM 基线训练
├── run_pipeline.py          # 数据管道 CLI（支持 --csv 参数）
│
├── ui/                      # Web 前端
│   ├── app.py               # 主页面 + 侧边栏
│   ├── i18n.py              # 国际化（中/英双语）
│   ├── core/
│   │   └── service_adapter.py  # UI ↔ 后端桥接层
│   └── pages/
│       ├── 1-数据浏览与看板.py
│       ├── 2-模型训练与校准.py
│       ├── 3-系统配置管理.py
│       └── 4-手动预测.py
│
├── scripts/
│   └── diagnose.py          # 系统诊断工具
│
├── utils/
│   └── tft_dataset.py       # TFT 数据集工具
│
├── data/                    # 数据文件
│   └── scada_data.csv
│
├── models/                  # 训练好的模型
│   ├── industrial_tft.onnx  # TFT ONNX 模型
│   └── lgbm/
│       └── lgbm_wwtp.txt    # LightGBM 模型
│
├── artifacts/               # 运行时产物
│   ├── scaler.pkl           # 特征归一化器
│   ├── target_scaler.pkl    # 目标归一化器
│   └── pipeline_config.json # 管道配置快照
│
├── logs/                    # 运行日志
├── offline_packages/        # 离线安装包（120 个 .whl）
└── test/                    # 测试脚本
```

---

## 安全机制

### 模型文件完整性保护

系统在加载模型文件前自动进行**头部魔数校验**，防止损坏文件导致进程卡死：

| 文件类型 | 校验方式 |
|---------|---------|
| LGBM 模型 (`.txt`) | 检查文件头是否以 `tree` 开头 |
| ONNX 模型 (`.onnx`) | 检查 protobuf 魔数 `0x08` |
| joblib 归一化器 (`.pkl`) | 检查 pickle 协议头 + 文件大小 |

所有加载操作均有 **15 秒超时保护**，损坏文件会被立即拒绝并降级运行。

### LGBM 特征维度自动修复

当 artifacts 中的 `lgbm_base_columns` 丢失或过期时，系统会**从 LGBM 模型文件反推基础特征列**，自动重建特征构建器，无需手动重跑管道。

---

## 训练模型

```bash
# 激活环境
venv\Scripts\activate

# 数据准备（首次或更换数据源后必须运行）
python run_pipeline.py
# 指定数据源（无需修改 config.yaml）
python run_pipeline.py --csv data/my_plant.csv

# LightGBM 基线（~1 分钟）
python run_lgbm.py

# TFT 时序模型（~3 分钟）
python train_tft.py

# PPO 强化学习（~30 分钟+）
python train.py --total-timesteps 500000
```

也可以在 Web UI 的「模型训练与校准」页面一键提交训练任务，后台运行不阻塞。

---

## 配置说明

所有配置集中在 `config.yaml`，也可通过 Web UI 的「系统配置管理」页面在线修改。

| 配置块 | 说明 |
|--------|------|
| `paths` | 数据/模型/日志/产物目录 |
| `pipeline` | 采样频率、回溯窗口、物理限幅 |
| `model` | TFT 架构参数、融合权重、ONNX opset（≥18） |
| `training` | epochs、batch_size、学习率、早停 |
| `asm1` | ASM1 动力学参数、曝气/回流边界 |
| `lgbm` | LightGBM 特征工程配置 |
| `sensors` | 传感器列名映射、状态变量映射 |
| `rl` | 强化学习启发式规则 |

---

## 数据源选择

无需手动编辑 `config.yaml`，在 Web UI 的「模型训练与校准」→「数据准备」中：

- **下拉选择** `data/` 目录已有 CSV
- **拖拽上传** 新 CSV 文件
- **自定义路径** 手动输入

选择后点击「设为数据源」自动更新配置。

---

## 故障排查

### 启动失败

```bash
# 1. 运行诊断
venv\Scripts\python.exe scripts\diagnose.py

# 2. 手动启动查看完整错误
venv\Scripts\python.exe -m streamlit run ui/app.py --server.port=8501
```

### 页面卡死 / 模型加载失败

最常见的两个原因：

| 症状 | 原因 | 解决 |
|------|------|------|
| 页面加载后数秒卡死 | LGBM 模型文件损坏 | `python scripts/diagnose.py` 检查第 5 项 |
| 启动即无响应 | Python 版本不匹配 | 确认 Python 3.10 64-bit，`python --version` |

模型文件损坏时，系统会自动降级为单模型运行，不会崩溃。手动处理：

```bash
# 跳过损坏模型
ren models\lgbm\lgbm_wwtp.txt lgbm_wwtp.txt.corrupted

# 重新训练
python run_pipeline.py && python run_lgbm.py
```

### 离线安装失败

```bash
# 确认 Python 版本是 3.10（离线包 cp310）
python --version
# 应为 Python 3.10.x

# 手动检查离线包完整性
dir offline_packages\numpy*cp310*.whl
# 应至少显示一个文件
```

如果是其他 Python 版本（3.11/3.12/3.13），需要联网安装或切换到 Python 3.10。

### 编码错误（GBK / UnicodeDecodeError）

`install.bat` 已内置 `PYTHONUTF8=1` 环境变量修复。若仍有问题：

```bash
# 手动设置后重试
set PYTHONUTF8=1
pip install -r requirements.txt
```

### ONNX 导出失败

```
RuntimeError: No Previous Version of LayerNormalization exists
```

opset 版本过旧。确认 `config.yaml` 中 `model.onnx_opset_version` ≥ 18（PyTorch 2.5+ 要求）。

### LGBM 预测为空 / 手动预测 LGBM 栏无数据

```
症状：「LightGBM 基线 已加载」但 LGBM 预测 COD 显示 "—"
日志：LGBM feature count mismatch: got 38, expected 101
```

**原因**：`artifacts/pipeline_config.json` 中 `lgbm_base_columns_for_inference` 为 null，回退到 4 列 TFT 特征名，无法匹配 101 维模型。

**自动修复**：系统已内置从 LGBM 模型反推基础列的功能。重启 Streamlit 服务即可触发。

**手动修复**：`python run_pipeline.py` 重新生成 artifacts，然后重启服务。

### 未连接 / 网页打不开

```bash
# 1. 确认端口未被占用
netstat -ano | find ":8501"

# 2. 查看服务器控制台窗口的错误信息
# （start.bat 现在会保留控制台窗口）

# 3. 手动启动排查
venv\Scripts\python.exe -m streamlit run ui/app.py --server.port=8501
```

---

## 常见问题

**Q: 如何更换 SCADA 数据源？**
> Web UI → 模型训练与校准 → 数据准备 → 选择/上传文件 → 点击设为数据源

**Q: 离线包支持哪些 Python 版本？**
> cp310（Python 3.10）和 cp314（Python 3.14）。其他版本需在线安装，`install.bat` 会自动处理。

**Q: 新电脑安装后网页打不开？**
> 1. 运行诊断：`venv\Scripts\python.exe scripts\diagnose.py`
> 2. 手动启动查看错误：`venv\Scripts\python.exe -m streamlit run ui/app.py --server.port=8501`
> 3. 检查 `start.bat` 打开的控制台窗口中的日志
> 4. 确认 `data/scada_data.csv` 存在，`models/` 目录有模型文件

**Q: `install.bat` 第二步闪退？**
> 已修复。新版用文件路径遍历代替 `python --version` 检测，避免 Windows Store 存根干扰。

**Q: GPU 训练支持吗？**
> 支持。修改 `config.yaml` 中 `model.device` 为 `"cuda"`，并安装 CUDA 版 PyTorch。

**Q: ASM1 模型与 SCADA 数据不对齐？**
> 修改 `config.yaml` 中 `sensors.state_mapping` 和 `sensors.history_fields`。

**Q: 如何切换界面语言？**
> 侧边栏底部「🌐 语言 / Language」→ 点选中文或 English。
