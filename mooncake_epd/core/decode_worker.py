"""
Decode Worker - Decode 阶段

接收 Prefill 节点传输的 KV Cache，执行自回归解码生成文本。
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecodeOutput:
    """Decode 阶段输出"""
    generated_ids: torch.Tensor       # [batch, num_generated_tokens]
    generated_text: Optional[str]     # 生成的文本
    decode_time_ms: float             # 总解码时间
    tokens_per_second: float          # tokens/s
    metadata: Dict[str, Any]
    request_id: str


class DecodeWorker:
    """
    Decode Worker

    接收 Prefill 传来的 KV Cache，执行自回归解码。
    每次生成一个 token，直到遇到 EOS 或达到最大长度。
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        tokenizer=None,
        max_new_tokens: int = 256,
        eos_token_id: int = 2,
    ):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self._request_counter = 0

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> DecodeOutput:
        """
        自回归解码。

        Args:
            input_ids: [batch, 1] 当前 token（从 Prefill 的 logits 采样）
            kv_cache: 从 Prefill 传输的 KV Cache
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: top-p 采样阈值

        Returns:
            DecodeOutput: 生成的 token IDs 和文本
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        start_time = time.perf_counter()

        input_ids = input_ids.to(self.device)
        generated = [input_ids]

        for step in range(max_tokens):
            model_kwargs = {
                "input_ids": input_ids,
                "use_cache": True,
            }

            if kv_cache is not None and step > 0:
                model_kwargs["past_key_values"] = self._rebuild_past_key_values(kv_cache)

            outputs = self.model(**model_kwargs)

            if hasattr(outputs, "logits"):
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)

            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                kv_cache = self._update_kv_cache(kv_cache, outputs.past_key_values)

            if next_token.item() == self.eos_token_id:
                break

            input_ids = next_token

        generated_ids = torch.cat(generated, dim=1)
        decode_time_ms = (time.perf_counter() - start_time) * 1000

        num_generated = generated_ids.shape[1]
        tps = num_generated / (decode_time_ms / 1000) if decode_time_ms > 0 else 0

        generated_text = None
        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(
                generated_ids[0].tolist(), skip_special_tokens=True
            )

        self._request_counter += 1
        request_id = f"decode_{self._request_counter}_{int(time.time() * 1000)}"

        output = DecodeOutput(
            generated_ids=generated_ids,
            generated_text=generated_text,
            decode_time_ms=decode_time_ms,
            tokens_per_second=tps,
            metadata={
                "num_generated_tokens": num_generated,
                "eos_reached": next_token.item() == self.eos_token_id,
            },
            request_id=request_id,
        )

        logger.info(
            f"Decode {request_id}: tokens={num_generated}, "
            f"time={decode_time_ms:.2f}ms, tps={tps:.1f}"
        )

        return output

    def _rebuild_past_key_values(self, kv_cache):
        """重建 past_key_values 格式"""
        key_cache, value_cache = kv_cache
        past = []
        for i in range(key_cache.shape[0]):
            past.append((key_cache[i], value_cache[i]))
        return past

    def _update_kv_cache(self, kv_cache, new_past_key_values):
        """增量更新 KV Cache"""
        if kv_cache is None:
            key_caches = [layer[0] for layer in new_past_key_values]
            value_caches = [layer[1] for layer in new_past_key_values]
            return (torch.stack(key_caches), torch.stack(value_caches))

        old_key, old_value = kv_cache
        new_keys = []
        new_values = []
        for i, (new_k, new_v) in enumerate(new_past_key_values):
            merged_k = torch.cat([old_key[i], new_k], dim=-2)
            merged_v = torch.cat([old_value[i], new_v], dim=-2)
            new_keys.append(merged_k)
            new_values.append(merged_v)
        return (torch.stack(new_keys), torch.stack(new_values))


class MockLLMForDecode(nn.Module):
    """
    模拟 LLM Decode，用于测试 EPD 流水线。
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 4096,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        num_attention_heads: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_attention_heads
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, use_cache=True, past_key_values=None, **kwargs):
        batch_size = input_ids.shape[0]

        logits = torch.randn(
            batch_size, 1, self.vocab_size,
            device=input_ids.device, dtype=torch.float32
        )

        kv_cache = []
        for _ in range(self.num_layers):
            k = torch.randn(
                batch_size, self.num_kv_heads, 1, self.head_dim,
                device=input_ids.device
            )
            v = torch.randn(
                batch_size, self.num_kv_heads, 1, self.head_dim,
                device=input_ids.device
            )
            kv_cache.append((k, v))

        class Output:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return Output(logits, kv_cache)


def create_decode_worker(
    device: torch.device,
    tokenizer=None,
    model: Optional[nn.Module] = None,
    **model_kwargs,
) -> DecodeWorker:
    """创建 Decode Worker"""
    if model is None:
        model = MockLLMForDecode(**model_kwargs)
    model = model.to(device).eval()
    return DecodeWorker(
        model=model, device=device, tokenizer=tokenizer
    )
