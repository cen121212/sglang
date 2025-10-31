from abc import ABC, abstractmethod

import torch
import torch_npu





from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.moe.token_dispatcher import DeepEPNormalDispatchOutput

COMM_STREAM = None
import torch.distributed as dist


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:

        a2a_out = torch.empty_like(input_)
    else:

        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:

        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


class MoETokenDispatcher(ABC):

    def __init__(self, **kwargs) -> None:
        


        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)
    
    @property
    def ep_group(self):

        return get_tp_group().device_group
    
    @property
    def ep_rank(self):
        return get_tensor_model_parallel_rank()
    
    @property
    def ep_size(self):
        return get_tensor_model_parallel_world_size()
    
    @abstractmethod
    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        raise NotImplementedError("Dispatch function not implemented.")
    
    @abstractmethod
    def token_combine(self, hidden_states: torch.Tensor, bias: torch.Tensor=None):
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithAll2AllV(MoETokenDispatcher):






    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_quant = False
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        self.hidden_shape = None
        self.topk_weights = None
        self.input_splits = None
        self.output_splits = None
        self.hidden_shape_before_permute = None



        self.num_global_tokens_per_local_expert = None


        self.tokens_per_experts = None
        self.global_input_tokens_local_experts_indices = None

        assert self.num_local_experts > 0, "Expected at least one experts"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )
        
        local_expert_indices_offset = self.ep_rank * self.num_local_experts

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert(
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices)-1):
            assert(
                self.local_expert_indices[i] == self.local_expert_indices[i+1] -1
            ), "local_expert_indices must be continous"
    
    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):

        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        assert topk_weights.dim() == 2, "Expected 20 tensor for topk_weights"
        assert topk_ids.dim() == 2, "Expected 20 tensor for routing map"




        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)
        self.reversed_local_input_permutation_mapping = (
            reversed_local_input_permutation_mapping
        )
        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens
            )

            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = (
                async_all_to_all(
                    dynamic_scale, 
                    self.output_splits, 
                    self.input_splits, 
                    self.ep_group
                )
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, 
            self.output_splits, 
            self.input_splits,
            self.ep_group,
            )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        global_input_tokens, dynamic_scale = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all
        )
        res = DeepEPNormalDispatchOutput(
            hidden_states=global_input_tokens,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states_scale=dynamic_scale,
            num_recv_tokens_per_expert=tokens_per_expert.tolist(),
        )
        return res

    def token_combine(self, hidden_states: torch.Tensor, bias: torch.Tensor = None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        hidden_states = self._combine_preprocess(hidden_states)



        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits, self.ep_group
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        output = self._combine_postprocess(permutated_local_input_tokens)


        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None
        self.topk_weights = None
        self.reversed_local_input_permutation_mapping = None
        self.reversed_global_input_permutation_mapping = None
        self.global_input_tokens_local_experts_indices = None

        return output
    
    def _dispatch_preprocess(self, hidden_states, topk_ids):
        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        tokens_per_expert = self._preprocess(topk_ids)

        self.hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = (
            torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=topk_ids,
                num_out_tokens=self.num_out_tokens,
            )
        )
        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
        )
    
    def _preprocess(self, topk_ids: torch.Tensor) -> torch.Tensor:
        num_local_tokens_per_expert = torch.histc(
            topk_ids, bins=self.num_experts, min=0, max=self.num_experts
        )

        ep_size = self.ep_size


        self.num_out_tokens = topk_ids.numel()




        self.input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = (
            get_tp_group()
            .all_gather(num_local_tokens_per_expert, dim=0)
            .reshape(ep_size, self.num_experts)
        )
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        if self.num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum."
            )
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(
            axis=0
        )






        if self.num_local_experts > 1:
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel(),
            )
        else:


            torch.npu.synchronize()
        
        return num_tokens_per_local_expert
    
    def _dispatch_postprocess(self, global_input_tokens, dynamic_scale=None):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, None
        
        # Handle quantized case
        if self.with_quant:
            assert (
                self.global_input_tokens_local_experts_indices is not None
            ), "global_input_tokens_local_experts_indices must be initalized before calling _dispatch_postprocess"
            expert_idx_2d = self.global_input_tokens_local_experts_indices.unsqueeze(-1)
            active_num = self.global_input_tokens_local_experts_indices.numel()


            if active_num <= 0:
                self.reversed_global_input_permutation_mapping = (
                    self.global_input_tokens_local_experts_indices
                )
                return global_input_tokens, dynamic_scale
            
            
            (
                global_input_tokens, 
                self.reversed_global_input_permutation_mapping,
                _, 
                expanded_scale
            ) = torch_npu.npu_moe_init_routing_v2(
                global_input_tokens,
                expert_idx_2d,
                scale=dynamic_scale,
                active_num=active_num,
                expert_capacity=0,
                expert_num=self.num_local_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_local_experts],
                quant_mode=-1,
                row_idx_type=0,
            )
            return global_input_tokens, expanded_scale
        

        global_input_tokens, self.reversed_global_input_permutation_mapping = (
            torch_npu.npu_moe_token_permute(
                global_input_tokens, self.global_input_tokens_local_experts_indices
            )
        )
        return global_input_tokens, None
    
    def _combine_preprocess(self, hidden_states):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping
            )

        return hidden_states
    
    def _combine_postprocess(self, permutated_local_input_tokens):
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=self.reversed_local_input_permutation_mapping.to(
                torch.int32
            ),
            probs=self.topk_weights,
            restore_shape=self.hidden_shape_before_permute,
        )

        
        output = output.view(self.hidden_shape)
        return output
        


        

