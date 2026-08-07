import joblib
from dataclasses import dataclass


@dataclass
class EdgeDeployConfig:
    model_path: str = "hyqad_model.joblib"
    quantization: bool = False
    half_precision: bool = False
    batch_size: int = 1
    use_onnx: bool = False


class ModelExporter:
    """
    Exports trained models for edge deployment on constrained
    devices (e.g., Jetson Nano). Supports serialization,
    quantization, and ONNX-like export patterns.
    """

    def __init__(self, deploy_config: EdgeDeployConfig | None = None):
        self.config = deploy_config or EdgeDeployConfig()

    def save(self, pipeline, path: str | None = None):
        path = path or self.config.model_path
        joblib.dump(pipeline, path, compress=3)
        return path

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def export_torch_script(self, model, example_input):
        try:
            import torch
            scripted = torch.jit.script(model)
            scripted.save(self.config.model_path.replace(".joblib", ".pt"))
            return True
        except Exception as e:
            print(f"TorchScript export failed: {e}")
            return False
