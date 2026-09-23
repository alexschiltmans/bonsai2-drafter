"""Rename a DFlash 2 drafter's two selector codebooks between the conventions in circulation.

The DFlash 2 candidate selector has two codebooks, `[vocab, rank]` each. The same tensors are
published under different names:

    zlab        candidate_selector.predecessor_codebook          z-lab's release; mlx-dspark
                candidate_selector.successor_codebook
    embedding   candidate_selector.predecessor_codebook.weight   an nn.Embedding's naming
                candidate_selector.successor_codebook.weight

A loader that checks names strictly refuses the other convention. This tool rewrites the two
names in the safetensors header and nothing else. Tensor bytes, dtypes, shapes and offsets are
untouched, and the data section is hashed before and after to prove it. A quantized codebook
(`.scales`/`.biases` beside `.weight`) cannot be expressed in the bf16 conventions and is refused.

    python3 scripts/rename-codebooks.py SRC_DIR DST_DIR --to zlab|embedding
    python3 scripts/rename-codebooks.py --self-test

SRC_DIR is never modified. DST_DIR must not exist. When the renamed header fits in the old
header's length, the weights are cloned (copy-on-write where the filesystem supports it) and
only the header is rewritten; otherwise the data section is streamed after a new header.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

CODEBOOKS = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
WEIGHTS = "model.safetensors"


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        return length, json.loads(stream.read(length))


def data_digest(path: Path, header_length: int) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(8 + header_length)
        for block in iter(lambda: stream.read(1 << 22), b""):
            checksum.update(block)
    return checksum.hexdigest()


def renamed(header: dict, to: str) -> dict:
    if any(f"{name}.scales" in header for name in CODEBOOKS):
        raise SystemExit("quantized codebooks (.scales/.biases) cannot be renamed losslessly")
    table = ({f"{name}.weight": name for name in CODEBOOKS} if to == "zlab"
             else {name: f"{name}.weight" for name in CODEBOOKS})
    present = [old for old in table if old in header]
    if not present:
        already = [new for new in table.values() if new in header]
        raise SystemExit(f"already in the {to} convention" if already
                         else "no DFlash 2 selector codebooks found")
    if len(present) != len(table):
        raise SystemExit(f"only some codebooks present: {present}")
    return {table.get(key, key): value for key, value in header.items()}


def clone(source: Path, destination: Path) -> None:
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(source), str(destination)],
                                                   check=False).returncode == 0:
        return
    shutil.copyfile(source, destination)


def rename(source_dir: Path, destination_dir: Path, to: str) -> dict:
    if destination_dir.exists():
        raise SystemExit(f"{destination_dir} exists; refusing to overwrite")
    source = source_dir / WEIGHTS
    length, header = read_header(source)
    new_header = renamed(header, to)
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    before = data_digest(source, length)
    staging = Path(tempfile.mkdtemp(prefix=".rename-", dir=destination_dir.parent))
    try:
        for item in source_dir.iterdir():
            if item.is_file() and item.name != WEIGHTS:
                shutil.copy2(item, staging / item.name)
        target = staging / WEIGHTS
        if len(blob) <= length:
            clone(source, target)
            with target.open("r+b") as stream:
                stream.seek(8)
                stream.write(blob + b" " * (length - len(blob)))
            new_length = length
        else:
            blob += b" " * (-len(blob) % 8)
            new_length = len(blob)
            with source.open("rb") as reader, target.open("wb") as writer:
                writer.write(struct.pack("<Q", new_length) + blob)
                reader.seek(8 + length)
                shutil.copyfileobj(reader, writer, 1 << 22)
        check_length, check_header = read_header(target)
        after = data_digest(target, check_length)
        if check_header != new_header or after != before:
            raise SystemExit("verification failed: the tensor data or header did not survive")
        os.rename(staging, destination_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"to": to, "header_bytes": [length, new_length], "data_sha256": before,
            "renamed": sorted(set(new_header) - set(header))}


def self_test() -> None:
    def write(path: Path, names: list[str]) -> None:
        header, data = {}, b""
        for index, name in enumerate(names):
            chunk = bytes([index + 1]) * 8
            header[name] = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [len(data), len(data) + 8]}
            data += chunk
        header["__metadata__"] = {"format": "mlx"}
        blob = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "zlab").mkdir()
        write(root / "zlab" / WEIGHTS, [*CODEBOOKS, "layers.0.weight"])
        (root / "zlab" / "config.json").write_text("{}")
        grown = rename(root / "zlab", root / "embedding", "embedding")      # header grows: stream
        assert grown["header_bytes"][1] >= grown["header_bytes"][0]
        _, header = read_header(root / "embedding" / WEIGHTS)
        assert all(f"{name}.weight" in header for name in CODEBOOKS) and "layers.0.weight" in header
        assert (root / "embedding" / "config.json").read_text() == "{}"
        back = rename(root / "embedding", root / "again", "zlab")           # header shrinks: clone
        assert back["data_sha256"] == grown["data_sha256"]
        assert read_header(root / "again" / WEIGHTS)[1] == read_header(root / "zlab" / WEIGHTS)[1]
        for bad, message in ((root / "zlab", "already"),):
            try:
                rename(bad, root / "x", "zlab")
            except SystemExit as error:
                assert message in str(error), error
            else:
                raise AssertionError("a no-op rename was accepted")
        try:
            rename(root / "zlab", root / "embedding", "embedding")
        except SystemExit as error:
            assert "exists" in str(error)
        else:
            raise AssertionError("an existing destination was overwritten")
        assert not [p for p in root.iterdir() if p.name.startswith(".rename-")]
    print("rename-codebooks self-test: ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--to", choices=("zlab", "embedding"))
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.source and arguments.destination and arguments.to:
        print(json.dumps(rename(arguments.source, arguments.destination, arguments.to), indent=2))
    else:
        parser.error("give SOURCE DESTINATION --to, or --self-test")
