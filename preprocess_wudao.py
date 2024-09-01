"""预处理悟道数据集"""
import math
import multiprocessing as mp
import ijson
import numpy as np
from config import *
from encoder import Encoder

def preprocess(queue: mp.Queue, encoder: Encoder, id: int): # 把所有文字用<eos>隔开然后连接在一起
    print(f"Worker {id} started.")
    while not queue.empty():
        fname = queue.get()
        with open(fname, "rb") as f_in, open(fname + ".bin", "ba") as f_out:
            for d in ijson.items(f_in, "item"):
                text = d["title"] + "：" + d["content"]
                code = encoder.encode(text) + [SPECIAL_TOKENS_IDS["<eos>"]]
                array = np.array(code, dtype=np.int16)
                f_out.write(array.tobytes())
        print(f"Done preprocessing {fname}. {queue.qsize()} files left.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_wudao.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    encoder = Encoder.from_path("encoder.json")

    queue = mp.Queue()
    for fname in fnames:
        queue.put(fname)

    num_workers = mp.cpu_count()
    workers = []
    for i in range(num_workers):
        worker = mp.Process(target=preprocess, args=(queue, encoder, i))
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()
