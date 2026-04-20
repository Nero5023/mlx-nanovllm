"""Single-layer paged KV storage, independent of the attention implementation.

Each sequence owns exclusive physical pages from a shared pool. Clear a sequence
explicitly when it finishes. These objects are intended for one worker thread.
"""

from collections import deque

import mlx.core as mx
from mlx_lm.models.cache import create_attention_mask
from mlx_lm.models.base import create_causal_mask


class KVPagePool:
    """Fixed-capacity K/V tensors: [physical page, token slot, KV head, head dim]."""

    def __init__(
        self, *, num_pages: int, page_size: int, num_kv_heads: int,
        head_dim: int, dtype: mx.Dtype = mx.float32,
    ) -> None:
        for name, value in (('num_pages', num_pages), ('page_size', page_size),
                            ('num_kv_heads', num_kv_heads), ('head_dim', head_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if dtype not in (mx.float32, mx.float16, mx.bfloat16):
            raise ValueError('KV pages require float32, float16, or bfloat16.')
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)
        self._free_pages: deque[int] = deque(range(num_pages))
        self._allocated: set[int] = set()

    @property
    def free_page_count(self) -> int:
        return len(self._free_pages)

    @property
    def used_page_count(self) -> int:
        return len(self._allocated)

    def _allocate(self, count: int) -> list[int]:
        # Check capacity before changing the free list or the sequence's table.
        if count > self.free_page_count:
            raise MemoryError(f'KV pool needs {count} free pages; only {self.free_page_count} available.')
        pages = [self._free_pages.popleft() for _ in range(count)]
        self._allocated.update(pages)
        return pages

    def _release(self, pages: list[int]) -> None:
        if len(set(pages)) != len(pages) or not set(pages).issubset(self._allocated):
            raise ValueError('Cannot release duplicate or unallocated KV pages.')
        self._allocated.difference_update(pages)
        self._free_pages.extend(pages)


class PagedKVCache:
    """One sequence in one layer; block_table maps logical pages to pool page IDs."""

    def __init__(self, pool: KVPagePool) -> None:
        self.pool = pool
        self._block_table: list[int] = []
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def block_table(self) -> tuple[int, ...]:
        return tuple(self._block_table)

    def append(self, keys: mx.array, values: mx.array) -> None:
        """Append [tokens, KV heads, head dim], allocating only missing pages.

        Capacity exhaustion and input validation leave the sequence unchanged.
        Array writes are lazy MLX operations; this is not a custom in-place kernel.
        """
        if keys.ndim != 3 or keys.shape != values.shape:
            raise ValueError('Keys and values must have the same [tokens, KV heads, head dim] shape.')
        if keys.shape[1:] != (self.pool.num_kv_heads, self.pool.head_dim):
            raise ValueError('KV head count and head dimension must match the pool.')
        if keys.dtype != self.pool.dtype or values.dtype != self.pool.dtype:
            raise ValueError('Keys and values must use the pool dtype.')
        if keys.shape[0] == 0:
            return

        end = self._length + keys.shape[0]
        required_pages = (end + self.pool.page_size - 1) // self.pool.page_size
        new_pages = self.pool._allocate(required_pages - len(self._block_table))
        blocks = self._block_table + new_pages
        page_ids = mx.array([blocks[pos // self.pool.page_size]
                             for pos in range(self._length, end)], dtype=mx.int32)
        offsets = mx.array([pos % self.pool.page_size
                            for pos in range(self._length, end)], dtype=mx.int32)
        self.pool.keys[page_ids, offsets] = keys
        self.pool.values[page_ids, offsets] = values
        self._block_table = blocks
        self._length = end

    def slot_mapping(self) -> mx.array:
        """Map valid logical tokens to flattened physical slots for inspection/gather."""
        return mx.array([
            self._block_table[pos // self.pool.page_size] * self.pool.page_size
            + pos % self.pool.page_size for pos in range(self._length)
        ], dtype=mx.int32)

    def gather(self) -> tuple[mx.array, mx.array]:
        """Materialize contiguous valid K/V for reference checks, not paged attention."""
        shape = (self.pool.num_pages * self.pool.page_size,
                 self.pool.num_kv_heads, self.pool.head_dim)
        slots = self.slot_mapping()
        return self.pool.keys.reshape(shape)[slots], self.pool.values.reshape(shape)[slots]

    def clear(self) -> None:
        """Return owned pages to the pool; safe to repeat and to append afterward."""
        self.pool._release(self._block_table)
        self._block_table = []
        self._length = 0


class PagedKVAdapter:
    """Single-request MLX-LM cache interface backed by pages, then gather + SDPA.

    Allocate a request-owned pool on the first write so its dtype matches actual
    model activations, including quantized models. This is a reference path, not
    direct paged attention. Fixed batches share this pool via PagedBatchKVCache.
    """

    def __init__(self, *, num_pages: int, page_size: int) -> None:
        for value in (num_pages, page_size):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('num_pages and page_size must be positive integers.')
        self.num_pages = num_pages
        self.page_size = page_size
        self.storage: PagedKVCache | None = None

    @property
    def offset(self) -> int:
        return self.storage.length if self.storage is not None else 0

    @property
    def state(self) -> tuple[mx.array, ...]:
        if self.storage is None:
            return ()
        return self.storage.pool.keys, self.storage.pool.values

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None):
        return create_attention_mask(N, self.offset, return_array, window_size)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Accept [1, KV heads, new tokens, head dim]; gather the full valid history."""
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[0] != 1:
            raise ValueError('PagedKVAdapter requires matching single-request [1, heads, tokens, dim] K/V.')
        if keys.dtype != values.dtype:
            raise ValueError('Keys and values must have the same dtype.')
        self._get_pool(keys)
        assert self.storage is not None
        # Storage is token-major; Qwen3 attention uses head-major tensors.
        self.storage.append(keys[0].transpose(1, 0, 2), values[0].transpose(1, 0, 2))
        gathered_keys, gathered_values = self.storage.gather()
        return (gathered_keys.transpose(1, 0, 2)[None],
                gathered_values.transpose(1, 0, 2)[None])

    def clear(self) -> None:
        if self.storage is not None:
            self.storage.clear()

    def _get_pool(self, keys: mx.array) -> KVPagePool:
        if self.storage is None:
            self.storage = PagedKVCache(KVPagePool(
                num_pages=self.num_pages, page_size=self.page_size,
                num_kv_heads=keys.shape[1], head_dim=keys.shape[3], dtype=keys.dtype,
            ))
        return self.storage.pool


class PagedBatchKVCache:
    """One layer of a fixed batch: independent row tables share the engine's pool.

    Only real prompt tokens enter pages. Gather temporarily left-pads histories
    for existing attention; filter releases completed rows without moving KV.
    """

    def __init__(self, owner: PagedKVAdapter, left_padding: list[int]) -> None:
        self.owner = owner
        self.rows: list[PagedKVCache] = []
        self._padding = left_padding

    @property
    def offset(self) -> mx.array:
        return mx.array([row.length for row in self.rows] if self.rows
                        else [-pad for pad in self._padding])

    @property
    def _width(self) -> int:
        return max((row.length for row in self.rows), default=0)

    @property
    def left_padding(self) -> mx.array:
        return mx.array([self._width - row.length for row in self.rows] if self.rows else self._padding)

    @property
    def state(self) -> tuple[mx.array, ...]:
        return self.owner.state

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None):
        return create_causal_mask(N, offset=self._width, left_padding=self.left_padding,
                                  window_size=window_size)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        first = not self.rows
        if keys.ndim != 4 or keys.shape != values.shape or keys.dtype != values.dtype:
            raise ValueError('Batched K/V must have matching [batch, heads, tokens, dim] shape and dtype.')
        if keys.shape[0] != len(self._padding):
            raise ValueError('KV rows must match the current batch.')
        if first and any(pad < 0 or pad >= keys.shape[2] for pad in self._padding):
            raise ValueError('Each fixed-batch prompt must contain real tokens.')
        pool = self.owner._get_pool(keys)
        if first:
            self.rows = [PagedKVCache(pool) for _ in self._padding]
        starts = self._padding if first else [0] * len(self.rows)
        needed = sum((row.length + keys.shape[2] - start + pool.page_size - 1) // pool.page_size
                     - len(row.block_table) for row, start in zip(self.rows, starts, strict=True))
        if needed > pool.free_page_count:
            raise MemoryError('Insufficient KV pages for the batch update.')
        for i, (row, start) in enumerate(zip(self.rows, starts, strict=True)):
            row.append(keys[i, :, start:, :].transpose(1, 0, 2),
                       values[i, :, start:, :].transpose(1, 0, 2))
        gathered_keys, gathered_values = [], []
        width = self._width
        for row in self.rows:
            k, v = row.gather()
            padding = [(width - row.length, 0), (0, 0), (0, 0)]
            gathered_keys.append(mx.pad(k, padding).transpose(1, 0, 2))
            gathered_values.append(mx.pad(v, padding).transpose(1, 0, 2))
        return mx.stack(gathered_keys), mx.stack(gathered_values)

    def filter(self, keep_rows: list[int]) -> None:
        keep = set(keep_rows)
        for i, row in enumerate(self.rows):
            if i not in keep:
                row.clear()
        self.rows = [self.rows[i] for i in keep_rows]
        self._padding = [self._padding[i] for i in keep_rows]

    def clear(self) -> None:
        for row in self.rows:
            row.clear()
        self.rows = []
        self._padding = []
