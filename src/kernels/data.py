from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm


def encode(encoder: tiktoken.Encoding, out_path: Path, raw_path: Path, N: int = 1024 * 1024):
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
    total = bin_path.stat().st_size
    assert total % dtype.itemsize == 0, "File size not a multiple of dtype size"
    length = total // dtype.itemsize
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(length,))
