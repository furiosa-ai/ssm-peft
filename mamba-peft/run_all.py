import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import subprocess
from threading import Thread
from queue import Queue, Empty
from pathlib import Path
import torch
import time


def wait_gpu_free(id):
    free_count = 0
    free_count_req = 10
    sleep_sec = 30.0

    print(f"Waiting for GPU {id} to go idle...")
    while True:
        device_util = torch.cuda.utilization(id)

        if device_util == 0:
            free_count += 1
            print(f"GPU {id} free at {free_count}/{free_count_req}")

            if free_count >= free_count_req:
                return
        else:
            free_count = 0

        time.sleep(sleep_sec)


def worker_func(qu: Queue, device, wait_free=False):
    env =  os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    if wait_free:
        wait_gpu_free(int(device))

    while True:
        try:
            proc = qu.get(block=False)
        except Empty:
            return
        
        subprocess.run(proc, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--device", nargs="+", required=True)
    parser.add_argument("--wait_gpu_free", action="store_true")
    parser.add_argument("--reversed", action="store_true")
    # parser.add_argument("--cfg", nargs="+", required=True)
    args, other_args = parser.parse_known_args()


    try:
        split_idx = other_args.index("--")
        other_args_var, other_args_const = other_args[:split_idx], other_args[split_idx+1:]
    except ValueError:
        split_idx = next(((i+1) for i, arg in enumerate(other_args[1:]) if arg.startswith("--")), len(other_args))
        other_args_var, other_args_const = other_args[:split_idx], other_args[split_idx:]

    script = args.script
    devices = args.device

    assert Path(script).is_file()

    other_args_var_grouped = []
    for arg in other_args_var:
        if arg.startswith("--"):
            other_args_var_grouped.append([])
        other_args_var_grouped[-1].append(arg)

    num_args_var = len(other_args_var_grouped)
    num_tasks = len(other_args_var_grouped[0]) - 1

    cmds = []
    for i in (range(num_tasks) if not args.reversed else reversed(range(num_tasks))):
        cmd = ["python", script]

        for group in other_args_var_grouped:
            cmd += [group[0], group[i+1]]

        cmd += other_args_const
        cmds.append(cmd)

    print("\n".join([" ".join(c) for c in cmds]))
    print("Devices:", args.device)

    qu = Queue()
    for cmd in cmds:
        qu.put(cmd)

    workers = [Thread(target=worker_func, args=(qu, device, args.wait_gpu_free)) for device in devices]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


if __name__ == "__main__":
    main()
