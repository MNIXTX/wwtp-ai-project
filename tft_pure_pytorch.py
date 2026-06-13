# tft_pure_pytorch.py (最终修复版)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import os
import time
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from loguru import logger
from torch.utils.data import DataLoader  # ✅ [FIX] 必须导入以支持类型提示和测试

# ==========================================
# 1. 核心算法组件 (保持不变)
# ==========================================
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_size, output_size)
        self.gate_act = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(output_size)
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            ctx = self.context_fc(context)
            if x.dim() == 3 and ctx.dim() == 2:
                ctx = ctx.unsqueeze(1)
            elif x.dim() == 2 and ctx.dim() == 3:
                ctx = ctx.squeeze(1)
            hidden = hidden + ctx

        hidden = self.elu(hidden)
        out = self.fc2(hidden)
        out = self.dropout(out)
        gate = self.gate_act(self.gate(out))
        out = out * gate
        out = self.layernorm(out + self.skip_proj(x))
        return out

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.feature_grns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_features)
        ])

        self.layernorm = nn.LayerNorm(hidden_size)
        self.weight_grn = GatedResidualNetwork(num_features * hidden_size, hidden_size, num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, num_features, hidden_size = x.shape

        processed_features = []
        for i in range(self.num_features):
            feat_out = self.feature_grns[i](x[:, :, i, :])
            processed_features.append(feat_out)

        processed = torch.stack(processed_features, dim=2)

        flattened = processed.reshape(batch_size, seq_len, -1)
        logits = self.weight_grn(flattened)
        weights = self.softmax(logits / 1.5)

        weights_expanded = weights.unsqueeze(-1)
        combined = (processed * weights_expanded).sum(dim=2)
        combined = self.layernorm(combined)

        return combined, weights

class IndustrialTFT(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, lstm_layers: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.input_embeddings = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(num_features)])
        self.vsn = VariableSelectionNetwork(num_features, hidden_size, dropout)
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True, dropout=lstm_dropout)
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, 1, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_features = [self.input_embeddings[i](x[:, :, i:i+1]) for i in range(self.num_features)]
        embedded = torch.stack(embedded_features, dim=2)

        vsn_output, feature_weights = self.vsn(embedded)
        lstm_out, _ = self.lstm(vsn_output)

        prediction = self.output_grn(lstm_out[:, -1, :])
        return prediction, feature_weights

# ==========================================
# 2. 解耦核心：状态数据类与回调接口
# ==========================================
@dataclass
class TrainConfig:
    """训练配置数据类"""
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    seq_len: int = 24
    num_features: int = 4
    hidden_size: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1
    device: str = "auto"
    target_clip_min: Optional[float] = None
    target_clip_max: Optional[float] = None

@dataclass
class TrainingState:
    """训练状态载体"""
    epoch: int
    total_epochs: int
    loss: float
    lr: float
    elapsed_time: float
    feature_weights: Optional[np.ndarray] = None

ProgressCallback = Callable[[float, str], None]
StateCallback = Callable[[TrainingState], None]
LogCallback = Callable[[str, str], None]

# ==========================================
# 3. 算法引擎控制器
# ==========================================
class TFTEngine:
    """TFT 算法引擎"""

    _session_cache: Dict[str, ort.InferenceSession] = {}

    def __init__(self, config: TrainConfig,
                 progress_cb: Optional[ProgressCallback] = None,
                 state_cb: Optional[StateCallback] = None,
                 log_cb: Optional[LogCallback] = None):
        self.cfg = config
        self.progress_cb = progress_cb or (lambda p, t: None)
        self.state_cb = state_cb or (lambda s: None)
        self.log_cb = log_cb or (lambda lvl, msg: logger.log(lvl, msg))

        self._stop_requested = False

        dev_name = self.cfg.device
        self.device = torch.device(dev_name if torch.cuda.is_available() or dev_name == 'cpu' else 'cpu')
        self._log("INFO", f"🚀 引擎初始化 | 使用设备: {self.device}")

        self.model = IndustrialTFT(
            num_features=self.cfg.num_features,
            hidden_size=self.cfg.hidden_size,
            lstm_layers=self.cfg.lstm_layers,
            dropout=self.cfg.dropout
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def _log(self, level: str, msg: str):
        self.log_cb(level, msg)

    def request_stop(self):
        self._stop_requested = True
        self._log("WARNING", "⚠️ 收到停止指令，将在当前 Epoch 结束后安全终止...")

    # ✅ [核心修改] train 方法现在只接收 dataloader
    def train(self, dataloader: DataLoader) -> Dict[str, Any]:
        """执行训练流程"""
        self._stop_requested = False
        self.model.train()
        start_time = time.time()

        self._log("INFO", f"--- 开始训练 TFT ({self.cfg.epochs} Epochs) ---")
        self.progress_cb(0.0, "开始训练...")

        final_loss = 0.0
        final_weights = None

        for epoch in range(self.cfg.epochs):
            if self._stop_requested:
                self._log("WARNING", "🛑 训练已被用户手动终止。")
                break

            epoch_loss = 0.0
            num_batches = 0

            # ✅ 直接从 dataloader 获取批次数据
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                pred, weights = self.model(batch_X)

                # 目标值裁剪 (可选)
                if self.cfg.target_clip_min is not None or self.cfg.target_clip_max is not None:
                    pred = torch.clamp(pred, min=self.cfg.target_clip_min, max=self.cfg.target_clip_max)

                loss = self.criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                final_weights = weights

            avg_loss = epoch_loss / max(1, num_batches)
            final_loss = avg_loss

            state = TrainingState(
                epoch=epoch + 1,
                total_epochs=self.cfg.epochs,
                loss=avg_loss,
                lr=self.optimizer.param_groups[0]['lr'],
                elapsed_time=time.time() - start_time,
                feature_weights=final_weights.mean(dim=(0, 1)).detach().cpu().numpy() if final_weights is not None else None
            )
            self.state_cb(state)

            progress = (epoch + 1) / self.cfg.epochs * 80
            self.progress_cb(progress, f"Epoch {epoch+1}/{self.cfg.epochs} | Loss: {avg_loss:.4f}")

        self._log("SUCCESS", "✅ 训练流程结束。")
        return {"model": self.model, "final_loss": final_loss}

    def export_onnx(self, save_dir: str, filename: str = "industrial_tft.onnx") -> str:
        """导出 ONNX 模型"""
        self.progress_cb(85.0, "正在导出 ONNX 模型...")
        os.makedirs(save_dir, exist_ok=True)
        onnx_path = os.path.join(save_dir, filename)

        self.model.eval()
        dummy_input = torch.randn(1, self.cfg.seq_len, self.cfg.num_features).to(self.device)

        # [修复] 只将 batch_size 声明为动态维度，seq_len 和 num_features 保持静态
        # 因为模型在 trace 时已经固定了序列长度和特征数，声明动态会导致冲突
        dynamic_axes = {
            'input_data': {0: 'batch_size'},
            'prediction': {0: 'batch_size'},
            'feature_weights': {0: 'batch_size'}
        }

        # [修复] Windows 控制台 GBK 编码兼容：临时将 stdout 切换为 UTF-8
        # PyTorch ONNX 内部 _verbose_print 会打印 Unicode 字符 (如 ❌)，GBK 无法编码
        import io as _io
        _old_stdout = None
        try:
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and \
               sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
                _old_stdout = sys.stdout
                sys.stdout = _io.TextIOWrapper(
                    sys.stdout.buffer, encoding='utf-8',
                    errors='replace', line_buffering=sys.stdout.line_buffering
                )
        except (AttributeError, OSError):
            pass

        # --- 清理旧模型文件（防止残留 .onnx.data 导致写入冲突）---
        import glob as _glob
        for _old_file in _glob.glob(onnx_path + "*"):
            try:
                os.remove(_old_file)
                self._log("INFO", f"已清理旧模型文件: {_old_file}")
            except Exception:
                pass

        # --- 获取 ONNX opset 版本（PyTorch 2.5+ 最低要求 18）---
        from config_manager import CFG as _CFG
        _opset = max(getattr(_CFG.model, 'onnx_opset_version', 14), 18)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    self.model, dummy_input, onnx_path,
                    export_params=True,
                    opset_version=_opset,
                    input_names=['input_data'],
                    output_names=['prediction', 'feature_weights'],
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=False
                )
        finally:
            # 恢复原始 stdout
            if _old_stdout is not None:
                try:
                    sys.stdout = _old_stdout
                except Exception:
                    pass

        self._log("SUCCESS", f"ONNX export success: {onnx_path}")
        self.progress_cb(95.0, "ONNX export done, verifying...")
        return onnx_path

    @staticmethod
    def verify_onnx(onnx_path: str, seq_len: int, num_features: int) -> dict:
        """验证 ONNX 推理"""
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        test_input = np.random.randn(1, seq_len, num_features).astype(np.float32)
        ort_outs = session.run(None, {'input_data': test_input})
        return {
            "prediction": ort_outs[0][0][0],
            "weights_mean": ort_outs[1].mean(axis=(0,1)).tolist()
        }

    @classmethod
    def predict(cls, onnx_path: str, input_data: np.ndarray) -> float:
        """生产级推理接口"""
        if onnx_path not in cls._session_cache:
            cls._session_cache[onnx_path] = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            logger.info(f"💾 ONNX Session 已缓存: {onnx_path}")

        session = cls._session_cache[onnx_path]

        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)

        ort_outs = session.run(None, {'input_data': input_data})
        prediction_tensor = ort_outs[0]
        return float(prediction_tensor.flatten()[0])

# ==========================================
# 4. CLI 专属适配器 & 测试入口
# ==========================================
def cli_state_handler(state: TrainingState):
    if state.epoch % max(1, state.total_epochs // 10) == 0:
        logger.info(f"Epoch {state.epoch}/{state.total_epochs} | Loss: {state.loss:.4f} | Time: {state.elapsed_time:.1f}s")

def cli_log_handler(level: str, msg: str):
    logger.log(level, msg)

if __name__ == "__main__":
    # ✅ [测试代码修复] 这里也必须使用 DataLoader 才能正确测试
    from torch.utils.data import TensorDataset

    cli_config = TrainConfig(
        epochs=50, batch_size=128, learning_rate=1e-3,
        seq_len=24, num_features=4, hidden_size=64, lstm_layers=2, dropout=0.1,
        target_clip_min=0.0, target_clip_max=100.0
    )

    engine = TFTEngine(
        config=cli_config,
        progress_cb=lambda p, t: logger.info(f"[{p:.1f}%] {t}"),
        state_cb=cli_state_handler,
        log_cb=cli_log_handler
    )

    # 模拟数据
    X_dummy = torch.randn(1000, cli_config.seq_len, cli_config.num_features)
    y_dummy = torch.randn(1000, 1)

    # ✅ 创建 Dataset 和 DataLoader (这才是正确的调用方式)
    dataset = TensorDataset(X_dummy, y_dummy)
    dataloader = DataLoader(dataset, batch_size=cli_config.batch_size, shuffle=True)

    # ✅ 传入 dataloader 而不是 X, y
    engine.train(dataloader)

    onnx_path = engine.export_onnx("./models")

    result = TFTEngine.verify_onnx(onnx_path, cli_config.seq_len, cli_config.num_features)
    logger.success(f"🎉 验证通过 | 预测值: {result['prediction']:.4f}")

    # 测试 predict 接口
    test_seq = np.random.randn(cli_config.seq_len, cli_config.num_features)
    scalar_pred = TFTEngine.predict(onnx_path, test_seq)
    logger.success(f"🎉 predict 接口测试通过 | 返回标量: {scalar_pred:.4f}")