# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.40.1",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "bitsandbytes>=0.43",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu121"
# url = "https://download.pytorch.org/whl/cu121"
# explicit = true
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu121" }
# ///
from __future__ import annotations

import logging
from typing import Any

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.openvla import OpenVLAModelServer
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class OpenVLAQuantizedModelServer(OpenVLAModelServer):
    """OpenVLA with bitsandbytes 8-bit / 4-bit quantization for 16 GB GPUs."""

    def __init__(
        self,
        *,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        if load_in_8bit == load_in_4bit:
            raise ValueError(
                "OpenVLAQuantizedModelServer requires exactly one of "
                f"load_in_8bit / load_in_4bit (got 8bit={load_in_8bit}, 4bit={load_in_4bit})"
            )
        super().__init__(**kwargs)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import transformers.modeling_utils as _tmu
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading OpenVLA (quantized) from %s on %s (8bit=%s, 4bit=%s)",
            self.model_path,
            self._device,
            self.load_in_8bit,
            self.load_in_4bit,
        )

        # transformers 4.40.1's from_pretrained unconditionally calls
        # dispatch_model(), which then calls .to(device) — fine on multi-GPU but
        # rejected on single-GPU quantized models. Patch for the load: walk
        # params/buffers and move CPU tensors to the GPU manually, leaving
        # bnb-quantized weights (already on GPU) alone.
        _cuda_target = "cuda:0"

        def _fixed_dispatch(model: Any, *args: Any, **kwargs: Any) -> Any:
            for _n, _p in model.named_parameters():
                if _p.device.type == "cpu":
                    _p.data = _p.data.to(_cuda_target)
            for _n, _b in model.named_buffers():
                if _b.device.type == "cpu":
                    _b.data = _b.data.to(_cuda_target)
            model.hf_device_map = {"": 0}
            return model

        _tmu.dispatch_model = _fixed_dispatch

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16

        self._model = AutoModelForVision2Seq.from_pretrained(self.model_path, **load_kwargs)
        # bnb placed weights during from_pretrained — no explicit .to(device).
        logger.info("OpenVLA model loaded (quantized).")

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import numpy as np
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        pil_image = self._preprocess_image(obs)
        task_description = obs.get("task_description", "")
        prompt = f"In: What action should the robot take to {task_description}?\nOut:"

        # bnb's compute dtype is fp16; matching inputs avoids a dtype mismatch.
        inputs = self._processor(prompt, pil_image).to(self._device, dtype=torch.float16)

        kwargs: dict[str, Any] = {"do_sample": False}
        if self.unnorm_key:
            kwargs["unnorm_key"] = self.unnorm_key

        action = self._model.predict_action(**inputs, **kwargs)
        action_arr = np.asarray(action, dtype=np.float32)
        action_arr[..., -1] = -np.sign(2 * action_arr[..., -1] - 1)
        return {"actions": action_arr}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(OpenVLAQuantizedModelServer)
