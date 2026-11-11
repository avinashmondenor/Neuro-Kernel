"""Bounded, opt-in workload generators for manual dashboard testing.

Run from the repository root:
  python scripts/run_test_workload.py cpu --seconds 15
  python scripts/run_test_workload.py memory --seconds 15 --memory-mb 96
  python scripts/run_test_workload.py disk --seconds 15 --disk-mb 32

These create temporary local load only. They do not modify project datasets.
"""
from __future__ import annotations
import argparse
import os
import tempfile
import time


def run_cpu(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    value = 1
    while time.monotonic() < deadline:
        value = (value * 1664525 + 1013904223) % (2**32)
    print("CPU workload completed; checksum:", value)


def run_memory(seconds: int, memory_mb: int) -> None:
    block = bytearray(memory_mb * 1024 * 1024)
    # Touch each page so the OS commits the allocation.
    for i in range(0, len(block), 4096):
        block[i] = 1
    print(f"Holding approximately {memory_mb} MiB for {seconds} seconds.")
    time.sleep(seconds)
    del block


def run_disk(seconds: int, disk_mb: int) -> None:
    total = disk_mb * 1024 * 1024
    chunk = b"NEUROKERNEL-MEMBER5-TEST\n" * 4096
    deadline = time.monotonic() + seconds
    written = 0
    with tempfile.NamedTemporaryFile(prefix="neurokernel_test_", delete=True) as f:
        while time.monotonic() < deadline and written < total:
            part = chunk[: min(len(chunk), total - written)]
            f.write(part)
            written += len(part)
        f.flush()
        os.fsync(f.fileno())
        print(f"Wrote {written / (1024 * 1024):.1f} MiB to a temporary file.")
    print("Temporary test file removed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["cpu", "memory", "disk"])
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--memory-mb", type=int, default=96)
    parser.add_argument("--disk-mb", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    if not 16 <= args.memory_mb <= 256:
        parser.error("--memory-mb must be between 16 and 256")
    if not 1 <= args.disk_mb <= 128:
        parser.error("--disk-mb must be between 1 and 128")
    print(f"Starting {args.scenario} workload for at most {args.seconds}s.")
    if args.scenario == "cpu":
        run_cpu(args.seconds)
    elif args.scenario == "memory":
        run_memory(args.seconds, args.memory_mb)
    else:
        run_disk(args.seconds, args.disk_mb)


if __name__ == "__main__":
    main()
