# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
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

"""Scheduled two-rank CPU regression for router gradients through the EP all-gather.

``GroupedExperts.forward`` all-gathers the per-token routing probabilities across
the expert-parallel group before dispatching tokens to local experts. Routing
probabilities participate in the main-loss gradient, so the gather must be
autograd-safe: a plain ``dist.all_gather`` detaches every gathered tensor and
silently leaves the router trainable only through auxiliary losses.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import nemo_automodel.components.moe.experts as experts_module
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    _AllGatherConcatVarlenFn,
    _ReduceScatterVarlenFn,
    _ScatterReduceVarlenFn,
)

_N_EXPERTS = 4
_TOP_K = 2
_DIM = 16
_MOE_INTER_DIM = 32
# Uneven per-rank token counts exercise the variable-length gather path.
_TOKENS_PER_RANK = (3, 2)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tiny_moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=_TOP_K,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_DIM,
        inter_dim=_MOE_INTER_DIM,
        moe_inter_dim=_MOE_INTER_DIM,
        norm_topk_prob=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic global batch shared by the reference and EP runs."""
    generator = torch.Generator().manual_seed(1234)
    num_tokens = sum(_TOKENS_PER_RANK)
    x = torch.randn(num_tokens, _DIM, generator=generator)
    router_logits = torch.randn(num_tokens, _N_EXPERTS, generator=generator)
    weights, indices = router_logits.softmax(dim=-1).topk(_TOP_K, dim=-1)
    token_mask = torch.ones(num_tokens, dtype=torch.bool)
    return x, weights, indices, token_mask


def _build_experts(config: MoEConfig) -> GroupedExperts:
    generator = torch.Generator().manual_seed(4321)
    experts = GroupedExperts(config)
    with torch.no_grad():
        experts.gate_and_up_projs.copy_(torch.randn(experts.gate_and_up_projs.shape, generator=generator) * 0.05)
        experts.down_projs.copy_(torch.randn(experts.down_projs.shape, generator=generator) * 0.05)
    return experts


def test_equal_length_all_gather_reuses_rank_major_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal-length EP ranks must not duplicate the full gathered activation."""
    gathered = torch.randn(4, _DIM)
    monkeypatch.setattr(experts_module, "_all_gather_rank_major", lambda *_args: gathered)
    monkeypatch.setattr(dist, "get_rank", lambda _group: 0)

    ctx = SimpleNamespace()
    actual = _AllGatherConcatVarlenFn.forward(ctx, torch.randn(2, _DIM), object(), [2, 2], 2)

    assert actual is gathered


def _reference_forward_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-process (ep_size=1) forward/backward as ground truth."""
    experts = _build_experts(_tiny_moe_config())
    x, weights, indices, token_mask = _global_inputs()
    weights = weights.clone().requires_grad_(True)
    y = experts(x, token_mask, weights, indices)
    y.sum().backward()
    assert weights.grad is not None
    assert experts.gate_and_up_projs.grad is not None
    assert experts.down_projs.grad is not None
    return (
        y.detach(),
        weights.grad.detach(),
        experts.gate_and_up_projs.grad.detach(),
        experts.down_projs.grad.detach(),
    )


def _ep_router_grad_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        y_ref, weights_grad_ref, gate_up_grad_ref, down_grad_ref = _reference_forward_backward()

        ep_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        experts = _build_experts(_tiny_moe_config())
        experts.gate_and_up_projs = nn.Parameter(
            distribute_tensor(experts.gate_and_up_projs.detach(), ep_mesh, [Shard(0)])
        )
        experts.down_projs = nn.Parameter(distribute_tensor(experts.down_projs.detach(), ep_mesh, [Shard(0)]))
        experts_module._MAX_GROUPED_MM_ROWS = 2

        x, weights, indices, token_mask = _global_inputs()
        start = sum(_TOKENS_PER_RANK[:rank])
        end = start + _TOKENS_PER_RANK[rank]
        local_weights = weights[start:end].clone().requires_grad_(True)

        y_local = experts(x[start:end], token_mask[start:end], local_weights, indices[start:end])
        torch.testing.assert_close(y_local, y_ref[start:end], rtol=1e-4, atol=1e-5)

        y_local.sum().backward()

        # Pre-fix, the routing weights were gathered with a non-differentiable
        # ``dist.all_gather`` and the local router leaf received no gradient.
        assert local_weights.grad is not None, "router weights received no gradient through the EP all-gather"
        torch.testing.assert_close(local_weights.grad, weights_grad_ref[start:end], rtol=1e-4, atol=1e-5)

        experts_per_rank = _N_EXPERTS // world_size
        expert_start = rank * experts_per_rank
        expert_end = expert_start + experts_per_rank
        for actual_grad, expected_grad in (
            (experts.gate_and_up_projs.grad, gate_up_grad_ref[expert_start:expert_end]),
            (experts.down_projs.grad, down_grad_ref[expert_start:expert_end]),
        ):
            assert actual_grad is not None
            if isinstance(actual_grad, DTensor):
                actual_grad = actual_grad.to_local()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _reduce_scatter_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Force multiple feature chunks with a tiny tensor so this test covers
        # the bounded NCCL-count path used by long EP128 sequences.
        experts_module._MAX_EP_COLLECTIVE_NUMEL = 12
        first_chunk = torch.full((3, 4), float(rank + 1))
        second_chunk = torch.full((2, 4), float(10 * (rank + 1)))
        leaf = torch.cat([first_chunk, second_chunk]).requires_grad_(True)
        reduced = _ReduceScatterVarlenFn.apply(leaf * 1.0, dist.group.WORLD, [3, 2], 3)

        expected_value = 3.0 if rank == 0 else 30.0
        torch.testing.assert_close(reduced, torch.full_like(reduced, expected_value))

        rank_loss_scale = float(rank + 1)
        (reduced * rank_loss_scale).sum().backward()
        assert leaf.grad is not None
        expected_grad = torch.cat([torch.ones((3, 4)), torch.full((2, 4), 2.0)])
        torch.testing.assert_close(leaf.grad, expected_grad)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _scatter_reduce_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Force multiple feature chunks while covering both grouped-expert
        # output layouts, uneven per-rank token counts, and duplicate rows.
        experts_module._MAX_EP_COLLECTIVE_NUMEL = 12
        features = 5
        for rows_per_token in (1, _TOP_K):
            total_rows = sum(_TOKENS_PER_RANK) * rows_per_token
            destination_rows = torch.tensor(
                [0, 2, 4, 4] if rank == 0 else [1, 3, total_rows - 1],
                dtype=torch.long,
            ).remainder_(total_rows)
            generator = torch.Generator().manual_seed(100 + rank + rows_per_token)
            source = torch.randn(destination_rows.numel(), features, generator=generator, requires_grad=True)

            expected_global = torch.zeros(total_rows, features)
            expected_global.scatter_add_(0, destination_rows[:, None].expand_as(source), source.detach())
            dist.all_reduce(expected_global)

            actual = _ScatterReduceVarlenFn.apply(
                source,
                destination_rows,
                dist.group.WORLD,
                list(_TOKENS_PER_RANK),
                max(_TOKENS_PER_RANK),
                rows_per_token,
            )
            local_start = sum(_TOKENS_PER_RANK[:rank]) * rows_per_token
            local_rows = _TOKENS_PER_RANK[rank] * rows_per_token
            expected = expected_global[local_start : local_start + local_rows]
            if rows_per_token > 1:
                expected = expected.reshape(_TOKENS_PER_RANK[rank], rows_per_token, features)
            torch.testing.assert_close(actual, expected)

            actual.square().sum().backward()
            assert source.grad is not None
            torch.testing.assert_close(source.grad, 2 * expected_global[destination_rows])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _chunked_extreme_skew_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        x, weights, _, token_mask = _global_inputs()
        # Route every token to rank 0's experts. Rank 1 must still participate
        # in every empty-source collective round.
        indices = torch.tensor([[0, 1]]).expand(x.shape[0], -1).clone()
        reference = _build_experts(_tiny_moe_config())
        with torch.no_grad():
            expected = reference(x, token_mask, weights, indices)

        ep_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        distributed = _build_experts(_tiny_moe_config())
        distributed.gate_and_up_projs = nn.Parameter(
            distribute_tensor(distributed.gate_and_up_projs.detach(), ep_mesh, [Shard(0)])
        )
        distributed.down_projs = nn.Parameter(distribute_tensor(distributed.down_projs.detach(), ep_mesh, [Shard(0)]))
        experts_module._MAX_GROUPED_MM_ROWS = 2
        start = sum(_TOKENS_PER_RANK[:rank])
        end = start + _TOKENS_PER_RANK[rank]

        with torch.no_grad():
            actual = distributed(x[start:end], token_mask[start:end], weights[start:end], indices[start:end])

        torch.testing.assert_close(actual, expected[start:end], rtol=1e-4, atol=1e-5)

        reference.zero_grad(set_to_none=True)
        distributed.zero_grad(set_to_none=True)
        reference_weights = weights.clone().requires_grad_(True)
        expected = reference(x, token_mask, reference_weights, indices)
        expected.sum().backward()
        local_weights = weights[start:end].clone().requires_grad_(True)
        actual = distributed(x[start:end], token_mask[start:end], local_weights, indices[start:end])
        actual.sum().backward()

        assert reference_weights.grad is not None
        assert local_weights.grad is not None
        torch.testing.assert_close(actual, expected[start:end], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(local_weights.grad, reference_weights.grad[start:end], rtol=1e-4, atol=1e-5)

        experts_per_rank = _N_EXPERTS // world_size
        expert_start = rank * experts_per_rank
        expert_end = expert_start + experts_per_rank
        for actual_param, reference_param in (
            (distributed.gate_and_up_projs, reference.gate_and_up_projs),
            (distributed.down_projs, reference.down_projs),
        ):
            assert actual_param.grad is not None
            assert reference_param.grad is not None
            actual_grad = actual_param.grad.to_local() if isinstance(actual_param.grad, DTensor) else actual_param.grad
            torch.testing.assert_close(
                actual_grad,
                reference_param.grad[expert_start:expert_end],
                rtol=1e-4,
                atol=1e-5,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ep_all_gather_propagates_router_weight_gradients():
    mp.spawn(
        _ep_router_grad_worker, args=(len(_TOKENS_PER_RANK), _free_port()), nprocs=len(_TOKENS_PER_RANK), join=True
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ep_reduce_scatter_handles_uneven_tokens_and_gradients():
    mp.spawn(
        _reduce_scatter_worker,
        args=(len(_TOKENS_PER_RANK), _free_port()),
        nprocs=len(_TOKENS_PER_RANK),
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ep_sparse_scatter_reduce_handles_layouts_duplicates_and_gradients():
    mp.spawn(
        _scatter_reduce_worker,
        args=(len(_TOKENS_PER_RANK), _free_port()),
        nprocs=len(_TOKENS_PER_RANK),
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ep_chunked_experts_handle_extreme_routing_skew():
    mp.spawn(
        _chunked_extreme_skew_worker,
        args=(len(_TOKENS_PER_RANK), _free_port()),
        nprocs=len(_TOKENS_PER_RANK),
        join=True,
    )
