# WWTP AI 智慧水务管控平台

污水处理厂智能优化控制系统 — ASM1 机理模型 + TFT 时序预测 + LightGBM 基线 + PPO 强化学习

---

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10/11 x64 | Windows 11 x64 |
| Python | 离线: 3.10 / 3.14<br>在线: 3.10 ~ 3.14 全系列 | 3.10（离线部署）/ 3.14（开发） |
| 包管理器 | **uv**（自动安装，也支持 pip 回退） | uv 0.11+ |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 5 GB（含离线包） | 10 GB |
| 网络 | 可完全离线运行 | 在线（自动更新依赖） |
| **项目路径** | **短路径（<100 字符）** | **如 `D:\WWTP_AI_System`** |

> ⚠️ **重要：项目路径长度限制**
>
> Windows 默认有 **MAX_PATH = 260 字符** 的限制。PyTorch 等包含深层目录树的包在长路径下安装时会失败：
> ```
> OSError: [Errno 2] No such file or directory
> ```
> **禁止将项目放置在以下位置：**
> - OneDrive 同步文件夹
> - 微信/QQ 接收文件目录（`xwechat_files\...`）
> - 桌面或下载文件夹的深层子目录
> - 任何路径总长超过 100 字符的位置
>
> **正确做法：** 将 `WWTP_AI_System` 文件夹直接放在盘符根目录下，如 `D:\WWTP_AI_System` 或 `C:\WWTP_AI_System`。
>
> `install.bat` 启动时会自动检测路径长度并在超过阈值时给出警告。

---

## 快速开始（新电脑）

### 方法一：一键安装（推荐）

```batch
双击 install.bat
```

脚本使用 **[uv](https://github.com/astral-sh/uv)**（Rust 编写的极速 Python 包管理器）自动完成：

1. 自动安装 uv（如未安装，联网时 ~5 秒）
2. 检测网络 → 在线/离线模式
3. 查找/安装 Python 3.10+（uv 可自动下载 Python）
4. 创建虚拟环境 `venv\`
5. 安装全部依赖（**在线 ~10 秒**，离线 ~3 分钟）
6. 验证所有模块

| 模式 | Python 版本 | 包管理器 | 说明 |
|------|------------|----------|------|
| 在线 | 3.10 ~ 3.14 | **uv** | 极速安装，uv 自动管理 Python 版本 |
| 在线 | 3.10 ~ 3.14 | pip 回退 | uv 不可用时自动降级 |
| 离线 | 3.10、3.14 | uv / pip | 从 `offline_packages\` 安装 |
| 离线 | 3.11/3.12/3.13 | — | 提示不支持，引导安装 Python 3.10 或联网 |

> **uv 不可用时** → `install.bat` 自动回退到 pip 模式（兼容旧环境）。

### 方法二：命令行安装

**使用 uv（推荐，10-60 秒）：**
```bash
uv venv
uv pip install -r requirements.txt
```

**使用 pip（传统，3-5 分钟）：**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**离线安装：**
```bash
uv pip install --no-index --find-links=offline_packages -r requirements.txt
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
├── install.bat              # 一键安装（uv + pip 回退）
├── install_pip.bat           # 旧版纯 pip 安装器（备份）
├── start.bat                # 启动 Streamlit 服务
├── stop.bat                 # 停止服务
├── build_exe.bat            # 打包为独立 EXE
├── launcher.py              # 启动器入口
├── config.yaml              # 全局配置文件
├── requirements.txt         # Python 依赖清单
│
├── uninstall.py              # 一键卸载/重置工具
├── api_server.py             # (可选) FastAPI 推理服务
│
├── models/                   # 模型定义层
│   ├── tft.py                # TFT (VSN+LSTM+MultiHeadAttn+GLU)
│   ├── tft_pf.py             # (可选) pytorch-forecasting TFT 后端
│   ├── lgbm.py               # LightGBM + 双模型共识 + 时间序列CV
│   ├── lgbm_features.py      # 特征工程（滞后/滚动/VSL/循环编码）
│   └── asm1/                 # ASM1 机理模型
│       ├── ode.py            # ODE 求解器 (numba JIT + 温度校正)
│       ├── env.py            # PPO 强化学习环境
│       └── calibration.py    # 参数自动校准（差分进化）
│
├── pipeline/                 # 数据与推理管线
│   ├── data.py               # ETL + 滑动窗口 + 双Scaler
│   ├── inference.py          # 融合推理引擎 (ASM1+LGBM+TFT)
│   ├── gateway.py            # 预测网关 (ONNX+LGBM, 文件校验)
│   └── adapter.py            # 预测适配器 (UI桥接)
│
├── config/                   # 配置管理
│   └── manager.py            # Pydantic 校验 + 热重载
│
├── training/                 # 训练入口
│   ├── train_ppo.py          # PPO 策略训练 (SB3)
│   ├── train_tft.py          # TFT 模型训练
│   ├── train_lgbm.py         # LightGBM 基线训练
│   └── run_pipeline.py       # 数据管道 CLI (--csv 参数)
│
├── utils/                    # 工具集
│   ├── dataset.py            # TFT PyTorch Dataset
│   ├── diagnose.py           # 环境/模型诊断工具
│   ├── generate_data.py      # 物理真实数据生成 (ASM1仿真)
│   ├── tune_hyperparams.py   # Optuna 超参自动搜索
│   └── benchmark.py          # LGBM vs TFT vs Ensemble 基准
│
├── ui/                       # Streamlit 前端
│   ├── app.py                # 主页 + 侧边栏 + 系统状态
│   ├── init.py               # 公共初始化 (缓存/懒加载)
│   ├── i18n.py               # 中英文双语支持
│   ├── core/
│   │   ├── service_adapter.py # UI↔后端适配器
│   │   └── api_client.py     # FastAPI 推理客户端
│   └── pages/
│       ├── 1-数据浏览与看板.py
│       ├── 2-模型训练与校准.py
│       ├── 3-系统配置管理.py
│       └── 4-手动预测.py
│
├── data/                     # SCADA 数据文件
├── models/lgbm/              # 训练好的模型文件
├── logs/                     # 运行日志
├── artifacts/                # 运行时产物 (scaler/配置缓存)
└── offline_packages/         # 离线安装包
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
python training/run_pipeline.py
# 指定数据源（无需修改 config.yaml）
python training/run_pipeline.py --csv data/my_plant.csv

# LightGBM 基线（~1 分钟）
python training/train_lgbm.py

# TFT 时序模型（~3 分钟）
python training/train_tft.py

# PPO 强化学习（~30 分钟+）
python training/train_ppo.py --total-timesteps 500000
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

## 文件完整性校验

下载/拷贝项目后，在运行 `install.bat` 之前，建议先确认文件完整。

### 快速目视检查

| 必须存在 | 说明 |
|----------|------|
| `install.bat` | 一键安装脚本 |
| `start.bat` / `stop.bat` | 启动/停止脚本 |
| `requirements.txt` | Python 依赖清单 |
| `config.yaml` | 全局配置文件 |
| `launcher.py` | 程序入口 |
| `offline_packages\` 目录 | 离线安装包（~122 个 .whl 文件） |
| `models\` / `pipeline\` / `ui\` / `utils\` | 核心代码模块 |

### 命令行校验（PowerShell）

```powershell
# 统计文件数量（正常应为 ~43 个 .py + 4 个 .bat）
(Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch 'offline_packages|venv' }).Count
(Get-ChildItem -Filter *.bat).Count

# 检查离线包数量（正常为 122）
(Get-ChildItem offline_packages\*.whl).Count

# 检查关键文件是否存在
@("install.bat","start.bat","stop.bat","requirements.txt","config.yaml","launcher.py") | ForEach-Object {
    if (Test-Path $_) { Write-Host "[OK] $_" -ForegroundColor Green }
    else { Write-Host "[MISS] $_" -ForegroundColor Red }
}

# 运行系统诊断（安装完成后）
venv\Scripts\python.exe utils\diagnose.py
```

### 常见文件缺失原因

| 症状 | 原因 | 解决 |
|------|------|------|
| `.bat` 双击闪退 | 文件被压缩软件或网盘转为 LF 换行 | 重新解压（用 WinRAR/7-Zip，不要用中文版 WinRAR 的"解压到"） |
| `offline_packages\` 为空 | 未完整下载/拷贝 | 重新下载完整发布包 |
| 缺少 `venv\` | 尚未运行 `install.bat` | 先运行 `install.bat` |

---

## 故障排查

### 安装时 OSError: No such file or directory（路径过长）

```
ERROR: Could not install packages due to an OSError:
[Errno 2] No such file or directory: '...\torch\include\ATen\native\...'
```

**原因**：项目路径超过 Windows MAX_PATH（260 字符）限制。PyTorch 的 include 目录嵌套很深，对路径长度极其敏感。

**解决**：
1. **推荐**：把整个 `WWTP_AI_System` 文件夹移到短路径，如 `D:\WWTP_AI_System`
2. **备选**（需管理员 + 重启）：启用 Windows 长路径支持
   ```batch
   reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
   ```
3. 确认项目**不在** OneDrive、微信文件夹、桌面深层目录下

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
| 页面加载后数秒卡死 | LGBM 模型文件损坏 | `python utils/diagnose.py` 检查第 5 项 |
| 启动即无响应 | Python 版本不匹配 | 确认 Python 3.10 64-bit，`python --version` |

模型文件损坏时，系统会自动降级为单模型运行，不会崩溃。手动处理：

```bash
# 跳过损坏模型
ren models\lgbm\lgbm_wwtp.txt lgbm_wwtp.txt.corrupted

# 重新训练
python training/run_pipeline.py && python training/train_lgbm.py
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

**手动修复**：`python training/run_pipeline.py` 重新生成 artifacts，然后重启服务。

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

**Q: 安装时提示 "OSError: [Errno 2] No such file or directory"？**
> 路径太长，超过 Windows 260 字符限制。将项目文件夹移到 `D:\WWTP_AI_System` 或 `C:\WWTP_AI_System` 即可解决。详见上方[故障排查](#安装时-oserror-no-such-file-or-directory路径过长)。

**Q: 项目可以放在桌面或 OneDrive 里吗？**
> 不建议。OneDrive 路径极长（含 `xwechat_files\...`、中文用户名等），PyTorch 安装几乎必失败。桌面如果路径简短（如 `C:\Users\xxx\Desktop\WWTP_AI_System`）可能可以，但仍推荐放在盘符根目录。

**Q: uv 是什么？必须安装吗？**
> [uv](https://github.com/astral-sh/uv) 是 Rust 编写的极速 Python 包管理器，比 pip 快 10-100 倍。`install.bat` 会自动检测并安装 uv；如果 uv 不可用（如离线环境），脚本自动回退到传统 pip 模式。

**Q: `install.bat` 和 `install_pip.bat` 有什么区别？**
> `install.bat` 是新版，优先使用 uv（速度更快、错误信息更清晰），uv 不可用时自动退回到 pip。`install_pip.bat` 是旧版纯 pip 安装器的备份，功能完全相同，如需回退可使用。

**Q: 如何切换界面语言？**
> 侧边栏底部「🌐 语言 / Language」→ 点选中文或 English。
