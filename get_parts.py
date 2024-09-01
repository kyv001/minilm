"""将零碎的.bin文件拼接在一起并分割成10GB左右一块的部分"""
import os
from tqdm import tqdm

def get_parts(fnames: list[str], part_size: int=10*1024*1024*1024):
    current_size = 0
    current_fnames = []
    parts_fnames = []

    for fname in fnames:
        size = os.path.getsize(fname)
        current_size += size
        current_fnames.append(fname)
        if current_size >= part_size:
            print(f"{len(current_fnames)} files -> part size: {current_size}")
            parts_fnames.append(current_fnames)
            current_size = 0
            current_fnames = []

    if current_fnames:
        parts_fnames.append(current_fnames)

    return parts_fnames

def save_part(part_fnames: list[str], part_num: int):
    dirname = os.path.dirname(part_fnames[0])
    os.system(f"cat {' '.join(part_fnames)} > {os.path.join(dirname, f'part_{part_num}.bin')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python get_parts.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    parts_fnames = get_parts(fnames)
    for i, part_fnames in tqdm(enumerate(parts_fnames)):
        save_part(part_fnames, i)
