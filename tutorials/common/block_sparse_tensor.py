import torch
from torch.nn.attention.flex_attention import BlockMask

import common.utils as utils



class BlockSparseTensor:
    def __init__(self, physical: torch.Tensor, block_size: int, lookup_table: torch.Tensor, inverse_lookup_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        self.physical = physical
        self.block_size = block_size
        self.lookup_table = lookup_table
        self.inverse_lookup_indices = inverse_lookup_indices


    @staticmethod
    def from_broadcast(x: torch.Tensor, block_mask: BlockMask, batch_shape):
        x = utils.unify_batch_dimension(x, batch_shape)

        if x.dim() == 3:
            # Add explicit feature dimension
            x = x.unsqueeze(-1)
        
        if x.dim() != 4:
            raise ValueError('BlockSparseTensors can only be constructed from tensors with dimension 2 or 3, excluding batch dimensions.')

        batch_size, _, n_blocks = block_mask.kv_num_blocks.shape
        block_size = block_mask.BLOCK_SIZE[0]
        n_tokens = n_blocks * block_size

        x = x.expand(batch_size, n_tokens, n_tokens, -1)

        lookup_table = BlockSparseTensor._build_lookup_table(block_mask)
        inverse_indices = BlockSparseTensor._build_inverse_lookup_indices(block_mask, lookup_table)

        physical = x[inverse_indices]
        return BlockSparseTensor(physical, block_size, lookup_table, inverse_indices)


    @staticmethod
    def _build_inverse_lookup_indices(block_mask: BlockMask, lookup_table: torch.Tensor):
        batch_size, n_blocks, _ = lookup_table.shape
        kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
        block_size = block_mask.BLOCK_SIZE[0]
        total_num_blocks = torch.sum(kv_num_blocks_no_heads)
        device = block_mask.kv_num_blocks.device

        lookup_table = lookup_table.clone()
        kv_order = torch.argsort(block_mask.kv_indices[:, 0, :, :], dim=-1)
        invalids_mask = kv_order >= kv_num_blocks_no_heads[:, :, None]
        lookup_table[invalids_mask] = 1_000_000
        
        flattened_lookup = lookup_table.flatten()
        inverse_lookup = flattened_lookup.argsort()[:total_num_blocks]
        batch_idx, q_block_idx, k_block_idx = torch.unravel_index(inverse_lookup, (batch_size, n_blocks, n_blocks))

        batch_idx = batch_idx.reshape(-1, 1, 1).expand(-1, block_size, block_size)
        q_block_idx = q_block_idx.reshape(-1, 1, 1).expand(-1, block_size, block_size)
        k_block_idx = k_block_idx.reshape(-1, 1, 1).expand(-1, block_size, block_size)

        q_idx = torch.arange(block_size, device=device).reshape(1, block_size, 1).expand(total_num_blocks, -1, block_size)
        k_idx = torch.arange(block_size, device=device).reshape(1, 1, block_size).expand(total_num_blocks, block_size, -1)

        q_idx = q_block_idx * block_size + q_idx
        k_idx = k_block_idx * block_size + k_idx

        return batch_idx, q_idx, k_idx


    @staticmethod
    def _build_lookup_table(block_mask: BlockMask):
        batch_size, _, n_blocks = block_mask.kv_num_blocks.shape
        kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
        total_num_blocks = torch.sum(kv_num_blocks_no_heads)
        device = block_mask.kv_num_blocks.device

        bq_lookup = torch.nn.functional.pad(
            torch.cumsum(kv_num_blocks_no_heads.flatten()[:-1], dim=0),
            (1, 0)
        )
        bq_lookup = bq_lookup.reshape(batch_size, n_blocks, 1)

        k_impact = torch.argsort(block_mask.kv_indices[:, 0, :, :], dim=-1)


        lookup_table = bq_lookup + k_impact
        lookup_table = torch.clip(lookup_table, max=total_num_blocks-1)  # Clip to avoid out-of-bounds indices for invalid blocks
        
        return lookup_table


    def __getitem__(self, index):
        b, q, k, c = index
        W = self.block_size
        return self.physical[self.lookup_table[b, q//W, k//W], q%W, k%W, c]

    def _unwrap(self, other):
        if isinstance(other, BlockSparseTensor):
            return other.physical
        return other

    def _wrap(self, tensor):
        return BlockSparseTensor(tensor, self.block_size, self.lookup_table, self.inverse_lookup_indices)

    def __add__(self, other):
        return self._wrap(self.physical + self._unwrap(other))

    def __radd__(self, other):
        return self._wrap(self._unwrap(other) + self.physical)

    def __sub__(self, other):
        return self._wrap(self.physical - self._unwrap(other))

    def __rsub__(self, other):
        return self._wrap(self._unwrap(other) - self.physical)

    def __mul__(self, other):
        return self._wrap(self.physical * self._unwrap(other))

    def __rmul__(self, other):
        return self._wrap(self._unwrap(other) * self.physical)

    def __truediv__(self, other):
        return self._wrap(self.physical / self._unwrap(other))

    def __rtruediv__(self, other):
        return self._wrap(self._unwrap(other) / self.physical)

    def __pow__(self, other):
        return self._wrap(self.physical ** self._unwrap(other))

    def __neg__(self):
        return self._wrap(-self.physical)

    def __eq__(self, other):
        return self._wrap(self.physical == self._unwrap(other))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        bst = next(x for x in args if isinstance(x, BlockSparseTensor))

        def unwrap(x):
            return x.physical if isinstance(x, BlockSparseTensor) else x

        unwrapped_args = tuple(unwrap(a) for a in args)
        unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        result = func(*unwrapped_args, **unwrapped_kwargs)

        if isinstance(result, torch.Tensor):
            return bst._wrap(result)
        elif isinstance(result, tuple):
            return tuple(bst._wrap(r) if isinstance(r, torch.Tensor) else r for r in result)
        else:
            return result

    def clone(self):
        return self._wrap(self.physical.clone())

    def detach(self):
        return self._wrap(self.physical.detach())

    def requires_grad_(self, requires_grad=True):
        self.physical.requires_grad_(requires_grad)
        return self

    def to(self, *args, **kwargs):
        return self._wrap(self.physical.to(*args, **kwargs))

    @property
    def device(self):
        return self.physical.device

    @property
    def dtype(self):
        return self.physical.dtype

    def __repr__(self):
        return f"BlockSparseTensor(shape={tuple(self.physical.shape)}, device={self.device})"

