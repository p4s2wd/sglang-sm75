# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape across all runners.

    size: the per-phase capture size -- what the runner iterates over.
        Tokens for prefill/ragged verify; requests for ordinary decode.
        - prefill: num_tokens
        - decode:  bs
    stream_idx:   pdmux stream index, or None for single-stream runners.
    variant_label: optional execution variant (for example, "lora",
        "nolora", or "chunked_prefix"), or None for runners that don't
        record per-variant graphs.
    attention_variant: independent attention variant (DSA decode dual-graph
        "dense" / "sparse", candidate_*), or None when that dual-graph
        capture is not enabled. Composes with variant_label so LoRA and
        attention variants can be captured independently.
    seq_len_bucket: the KV width a decode graph scans. The indexer's logits row
        is sized from the page-table view, which is sliced with the max_seq_len
        fixed at capture time, so one graph captured at the full context length
        makes an 11-token request scan every column of that context on every
        step. Bucketing by length is worth ~1.5x on short-sequence decode.
        None means "not bucketed", which is what every runner other than the
        DeepSeek-V4 decode runner uses.
    """

    size: int
    # PDMux stream, or None for a single stream.
    stream_idx: Optional[int] = None
    # LoRA or prefill-prefix variant; None selects the default.
    variant_label: Optional[str] = None
    # Independent attention variant (DSA dense/sparse, candidate_*); None is default.
    attention_variant: Optional[str] = None
    seq_len_bucket: Optional[int] = None
