from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tiktoken
import torch
from jaxtyping import Integer
from tqdm import tqdm


def encode(
    encoder: tiktoken.Encoding,
    out_path: Path,
    raw_path: Path,
    N: int = 1024 * 1024,
    dtype: np.dtype = np.dtype(np.uint16),
) -> None:
    """Tokenize ``raw_path`` with ``encoder`` and write ``dtype`` tokens to ``out_path``.

    Args:
        encoder: tiktoken encoder used to convert UTF-8 text chunks into token IDs.
        out_path: Destination file that receives the serialized ``dtype`` token stream.
        raw_path: Source text file that is read progressively in chunks of size ``N``.
        N: Chunk size in bytes to read per iteration; helps limit peak memory usage.
        dtype: Data type used for serialized token IDs.
    """
    total = raw_path.stat().st_size
    total_tokens = 0
    with ExitStack() as s:
        f_in = s.enter_context(open(raw_path, encoding="utf-8"))
        f_out = s.enter_context(open(out_path, "w"))
        pbar = s.enter_context(tqdm(total=total, unit="B", unit_scale=True))

        while True:
            chunk = f_in.read(N)
            if not chunk:
                break
            tokens = encoder.encode_to_numpy(chunk, allowed_special=encoder.special_tokens_set).astype(np.uint16)
            total_tokens += len(tokens)
            tokens.tofile(f_out)
            bytes_read = f_in.tell()
            pbar.set_postfix({"output tokens": total_tokens}, refresh=False)
            pbar.update(bytes_read - pbar.n)


def open_mmap(bin_path: Path, dtype: np.dtype = np.dtype(np.uint16)) -> np.memmap:
    """Memory-map a binary token file without loading entire contents into memory.

    Args:
        bin_path: Path to the binary file containing serialized token IDs.
        dtype: Data type used for serialized token IDs.

    Returns:
        np.memmap: Array-like view backed by the file with shape ``(num_tokens,)``.
    """
    total = bin_path.stat().st_size
    assert total % dtype.itemsize == 0, "File size not a multiple of dtype size"
    length = total // dtype.itemsize
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(length,))


class Dataset(torch.utils.data.Dataset):
    """Memory-mapped token dataset that yields sequential training samples.

    Args:
        seq_len: Number of tokens returned per example.
        path: Path to the binary token file produced by ``encode``.
    """

    def __init__(self, seq_len: int, path: Path):
        super().__init__()
        self.seq_len = seq_len
        self.path = path
        # Lazy load to avoid worker processes loading array into memory when pickling Dataset.
        self._data = None

    @property
    def data(self):
        """The memory-mapped underlying token file."""
        if self._data is None:
            self._data = open_mmap(self.path)
        return self._data

    def __getitem__(
        self, idx: int
    ) -> tuple[Integer[torch.Tensor, "batch seq_len"], Integer[torch.Tensor, "batch seq_len"]]:
        """Return input/target token slices starting at index ``idx``."""
        x = torch.tensor(self.data[idx : idx + self.seq_len]).int()
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1]).int()
        return x, y

    def __len__(self) -> int:
        """Number of valid starting indices for length-``seq_len`` slices."""
        # Targets require `idx + 1:idx + seq_len + 1` to exist so effectively:
        # `len(self.data) - (self.seq_len + 1) + 1`
        return max(len(self.data) - self.seq_len, 0)
