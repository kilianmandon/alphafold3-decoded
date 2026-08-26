import torch
from torch.nn.attention.flex_attention import BlockMask

import common.utils as utils

class ExtendedBlockMask:
    """
    ExtendedBlockMask wraps a BlockMask from flex_attention to provide utilities for working with 
    block-sparse tensors. Block-sparse tensors represent a logical shape (batch_dim, q_dim, k_dim), 
    where q_dim and k_dim are potentially very large, but where the tensor is sparse and we only need to 
    store a small number of blocks (n_blocks, block_size, block_size). ExtendedBlockMask provides indices 
    to get the block_index in the flat layout based on a true logical index (batch_idx, q_idx, k_idx),
    and the inverse direction. Concretely, it provides:

    forward_index: Index to help getting a value based on a true logical index, e.g. (batch_idx, q_idx, k_idx),
      from a block-sparse tensor. forward_index has shape (batch_dim, n_blocks_q, n_blocks_k). It can 
      be queried to compute the index of the block within the flattened block layout that corresponds 
      to a specific (batch_idx, q_idx, k_idx) tuple, by forward_index[batch_idx, q_idx//block_size, k_idx//block_size].
    inverse_indices: A tuple of three indices, (batch_idx, q_idx, k_idx), each of shape 
      (total_num_blocks, block_size, block_size). These indices are able to extract the block representation
      from a full materialization of the logical tensor, e.g. for a full feature x of shape (batch_dim, q_dim, k_dim),
      we can extract x_blocks = x[batch_idx, q_idx, k_idx] to get the physical blocks in a flat layout.
    """

    def __init__(self, 
                block_mask: BlockMask,
                forward_index: torch.IntTensor,
                inverse_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        self.block_mask = block_mask
        self.forward_index = forward_index
        self.inverse_indices = inverse_indices

    @staticmethod
    def from_block_mask(block_mask: BlockMask):
        forward_index = ExtendedBlockMask._build_forward_index(block_mask)
        inverse_indices = ExtendedBlockMask._build_inverse_indices(block_mask, forward_index)

        return ExtendedBlockMask(block_mask, forward_index, inverse_indices)



    @staticmethod
    def _build_forward_index(block_mask: BlockMask):
        """
        Creates the forward index of shape (batch_dim, num_query_blocks, num_key_blocks). The forward index
        of a block at (batch, query_block, key_block) within the internal flat block layout. Concretely,
        physical[forward_index[batch, query_block, key_block]] should contain the data at index 
        (batch, query_block, key_block) in the logical model. 
        Args:
            block_mask (BlockMask): The BlockMask that contains the sparsity layout of the tensor.
            invalid_value (number, optional): The value that lookups at invalid (masked) locations 
              will lead to. For use of this forward index with FlexAttention, this needs to be a valid index
              in the physical layout. Defaults to 0. 

        Returns:
            torch.Tensor: Tensor of shape (batch_dim, num_query_blocks, num_key_blocks), that allows for lookup in 
              the internal flat block layout.
        """
        batch_size, _, n_blocks = block_mask.kv_num_blocks.shape
        kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
        total_num_blocks = torch.sum(kv_num_blocks_no_heads)
        kv_indices = block_mask.kv_indices[:, 0, :, :]

        forward_index = None

        """
        TODO: Build the forward index. You can do the computation based on kv_num_blocks and kv_indices. Here,
        kv_num_blocks is of shape (batch_dim, num_query_blocks), and contains the nuber of blocks in the block rows.
        It can be used to compute at which index in the flat block layout the blocks for a pair (b, q_block) start:
        If kv_num_blocks would be [ [2, 3], [1, 4]], the indices into the flat layout (based only on batch and query) 
        would be [ [0, 2], [5, 6] ], so basically a torch.cumsum with a leading zero and without the last value 
        (and with flattening and unflattening in between). Thus,

        Step 1: Compute bq_indices of shape (batch_dim, num_query_blocks), such that bq_indices[b, q] is the index
        where the blocks of the row (b, q) start in the flat layout.
        
        kv_indices can be used to also consider the key index. kv_indices is of shape 
        (batch_dim, num_query_blocks, num_key_blocks), and tells you which reordering of the key blocks is applied,
        before taking the first kv_num_blocks of them. Concretely, if at index (b, q), we had 
        kv_indices[b, q] = (3, 1, 6, 8, ...), and num_query_blocks[b, q] = 3, then the blocks 3, 1, and 6, are 
        non-masked in that row (and are the ones that are stored in the physical layout). Thus, we want that 
        forward_index[b, q, 3] = forward_index_batch_query + 0, forward_index[b, q, 1] = forward_index_batch_query + 1,
        forward_index[b, q, 6] = forward_index_batch_query + 2, and for all other key block indices k,
        forward_index[b, q, k] = ? does not matter (we just need to make sure that it is not out of bounds
        by clipping forward_index in the end, because flex_attention might query some of these, even if they get 
        masked out afterward).
        Here, forward_index_batch_query is the index into the flat layout where the blocks for this (b, q) start.
        You can achieve that by doing argsort on kv_indices: Since kv_indices are a permutation of (0, n_blocks-1), 
        for the example above, we'd have argsort(kv_indices[b, q]) = (x, 1, x, 0, x, x, 2, ...) =: kv_rank[b, q].
        So, basically, kv_rank[b, q, k] is the offset we need to add onto bq_indices (take care with the 
        shapes for broadcasting). So,

        Step 2: Compute kv_rank and build forward_index from bq_indices and kv_rank.

        Step 3: Invalid indices (b, q, k) that describe a block that's not present in the flat layout need to be 
        clipped to avoid illegal accesses within flex_attention. Use torch.clip so they don't exceed 
        total_num_blocks - 1.
        """

        bq_indices = torch.nn.functional.pad(
            torch.cumsum(kv_num_blocks_no_heads.flatten()[:-1], dim=0),
            (1, 0)
        )
        bq_indices = bq_indices.reshape(batch_size, n_blocks, 1)

        kv_rank = torch.argsort(kv_indices, dim=-1)


        forward_index = bq_indices + kv_rank
        forward_index = torch.clip(forward_index, max=total_num_blocks-1)  # Clip to avoid out-of-bounds indices for invalid blocks

        """ End of your code """
        
        return forward_index


    @staticmethod
    def _build_inverse_indices(block_mask: BlockMask, forward_index: torch.IntTensor):
        """
        This method computes so-called inverse indices, that can be used to turn a tensor that has the true logical
        shape, e.g. x with shape (batch_dim, num_queries, num_keys, c), into the physical layout: The corresponding
        data would be physical = x[batch_idx, q_idx, k_idx], where (batch_idx, q_idx, k_idx) is the return value 
        of this function. 

        Args:
            block_mask (BlockMask): The BlockMask describing the sparsity pattern in x.
            forward_index (torch.IntTensor): The forward index of the block mask, as provided by _build_forward_index.

        Returns:
            tuple: A tuple containing the following indices:
              batch_idx: IntTensor of shape (n_total_blocks, block_size, block_size), where batch_idx[n, :, :] is 
                constant and is the index of the batch that the n-th block in the flat layout is from.
              q_idx: IntTensor of shape (n_total_blocks, block_size, block_size), such that q_idx[n, q, k] is the 
                query index of the entry (q, k) in the n-th block in the flat layout (which basically means: 
                q_block_idx[n] * block_size + q).
              k_idx: IntTensor of shape (n_total_blocks, block_size, block_size), such that k_idx[n, q, k] is the 
                key index of the entry (q, k) in the n-th block in the flat layout (which basically means: 
                k_block_idx[n] * block_size + k).
        """
        forward_index = forward_index.clone()
        batch_size, n_blocks, _ = forward_index.shape
        kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
        block_size = block_mask.BLOCK_SIZE[0]
        total_num_blocks = torch.sum(kv_num_blocks_no_heads)
        device = block_mask.kv_num_blocks.device
        out_index_shape = (total_num_blocks, block_size, block_size)

        # We will set all indices in forward_index that belong to invalid (masked) blocks to a large number,
        # so that invalid blocks will be at the tail when argsorting the forward_index
        kv_order = torch.argsort(block_mask.kv_indices[:, 0, :, :], dim=-1)
        invalids_mask = kv_order >= kv_num_blocks_no_heads[:, :, None]
        forward_index[invalids_mask] = 1_000_000

        batch_idx, q_idx, k_idx = None, None, None

        """
        TODO: Compute batch_idx, q_idx, and k_idx. 
        This is basically the inverse of the forward_index. You can always think of the inverse of an indexing
        operation as the argsort of the indices. That's because for a permutation sigma, argsort(sigma) is the 
        inverse permutation. Concretely, if we computed B = A[[3, 0, 2, 1]], then 
        A = B[[1, 3, 2, 0]] = B[argsort([3, 0, 2, 1])]
        This works here as well: We can get inverse lookup indices by flattening the forward index 
        (forward_index is of shape (batch_dim, num_query_blocks, num_key_blocks)), then performing argsort,
        and truncating it to [:total_num_blocks] so that we only get the batch-query_block-key_block indices of valid
        blocks (because we set forward_index=1_000_000 for invalids, those will be trailing after argsort).
        The resulting inverse index is of shape (total_num_blocks,) and, because we flattened forward_index, it will
        satisfy inverse_index[n] = (flattened index of the block in the logical layout). That is, if inverse_index[n]
        points to the logical block (b: 2, q_block: 1, k_block: 2), and our logical shape is 
        (batch_dim: 5, q_block_dim: 4, k_block_dim: 3), then inverse_index[n] = 2 * (4*3) + 1 * (3) + 2.
        We can unravel this flat index into (2, 1, 2) using torch.unravel_index.

        After that, the hard work is done. The resulting batch, query_block, and key_block indices are of shape
        (total_num_blocks,) and map a flat block index to the corresponding logical block index. We just need to go
        from block indices to individual query indices by multiplying the query and key block indices by the block_size,
        and then adding range(block_size) along the correct dimensions (broadcasting!) to get to indices of 
        shape (total_num_blocks, block_size, block_size) that also respect the concrete query and key index 
        within the blocks.
        """
        
        flattened_lookup = forward_index.flatten()
        inverse_lookup = flattened_lookup.argsort()[:total_num_blocks]
        batch_idx, q_block_idx, k_block_idx = torch.unravel_index(inverse_lookup, (batch_size, n_blocks, n_blocks))

        batch_idx = batch_idx.reshape(-1, 1, 1).expand(out_index_shape)
        q_block_idx = q_block_idx.reshape(-1, 1, 1).expand(out_index_shape)
        k_block_idx = k_block_idx.reshape(-1, 1, 1).expand(out_index_shape)

        q_idx = torch.arange(block_size, device=device).reshape(1, -1, 1).expand(out_index_shape)
        k_idx = torch.arange(block_size, device=device).reshape(1, 1, -1).expand(out_index_shape)

        q_idx = q_block_idx * block_size + q_idx
        k_idx = k_block_idx * block_size + k_idx

        """ End of your code """

        return batch_idx, q_idx, k_idx


class BlockSparseTensor:
    """
    A BlockSparseTensor represents data in the layout of attention logits and mirrors the sparsity pattern of a 
    BlockMask from FlexAttention. Concretely, only blocks that contain unmasked values are stored in a flat block 
    layout internally, within the physical storage of shape (n_blocks, block_size, block_size), or 
    (n_blocks, block_size, block_size, feat_dim). 
    Note that BlockSparseTensor does not contain
    a dimension for the different attention heads that might be allowed in BlockMask. We can only represent values
    that are uniform along the attention-head dimension.
    """
    def __init__(self, physical: torch.Tensor, block_size: int, block_mask_with_metadata: ExtendedBlockMask):
        self.physical = physical
        self.block_size = block_size

        self.block_mask = block_mask_with_metadata
        self.forward_index = block_mask_with_metadata.forward_index
        self.inverse_indices = block_mask_with_metadata.inverse_indices


    @staticmethod
    def broadcast_up(x: torch.Tensor, extended_block_mask: ExtendedBlockMask, batch_shape):
        """
        This method takes a vector x of shape (**batch_shape, ...) that, after flattening the batch_shape to
        (batch_dim, ...), is broadcastable to (batch_dim, n_queries, n_keys, c). The method returns a BlockSparseTensor
        that is equivalent to actually broadcasting x to this full shape, but only stores the non-masked blocks.
        The channel dimension of x is optional.

        Args:
            x (torch.Tensor): A tensor of a shape that, after flattening the batch_shape, is broadcastable to
              (batch_dim, n_queries, n_keys, c). The channel dimension is optional. x will be broadcasted to this shape.
            extended_block_mask (ExtendedBlockMask): A block mask that shows the sparsity pattern of the mask to use.
            batch_shape (tuple): The batch shape of x.

        Raises:
            ValueError: This is raised if x has an incorrect number of dimensions.

        Returns:
            BlockSparseTensor: The result of the broadcasting operation.
        """
        x = utils.unify_batch_dimension(x, batch_shape)

        if x.dim() == 3:
            # Add explicit feature dimension
            x = x.unsqueeze(-1)
        
        if x.dim() != 4:
            raise ValueError('BlockSparseTensors can only be constructed from tensors with dimension 2 or 3, ' \
            'excluding batch dimensions.')

        block_size = extended_block_mask.block_mask.BLOCK_SIZE[0]
        inverse_indices = extended_block_mask.inverse_indices

        out_bst = None

        """
        TODO: Implement the broadcasting operation. The idea is this: If x already had the broadcasted shape,
        we could simply do physical = x[inverse_indices]. However, we don't want to realize that broadcasted shape 
        of x. The solution is to use x.expand(target_shape), which gives a view of x with the broadcasted shape,
        but without actually doing the memory duplication.
        """



        # First option: Expand x:
        # x = x.expand(batch_size, n_tokens, n_tokens, -1)
        # Second option: Clip indices along the dimension that's broadcasted to 0:
        inverse_indices = list(inverse_indices)
        for dim in [1, 2]:
            if x.shape[dim] == 1:
                inverse_indices[dim] = torch.zeros_like(inverse_indices[dim])
        inverse_indices = tuple(inverse_indices)

        physical = x[inverse_indices]
        out_bst = BlockSparseTensor(physical, block_size, extended_block_mask)

        """ End of your code """

        return out_bst

    # Utility functions to supprt some binary and unary operations
    def _unwrap(self, other):
        if isinstance(other, BlockSparseTensor):
            return other.physical
        return other

    def _wrap(self, tensor):
        return BlockSparseTensor(tensor, self.block_size, self.block_mask)

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

    def map(self, fn):
        return self._wrap(fn(self.physical))

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

