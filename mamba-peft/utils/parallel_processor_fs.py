

from multiprocessing import Process, Value
from pathlib import Path
import pickle
from tqdm import tqdm


class ParallelProcessorFS:
    def __init__(self, func, size, n, output_file) -> None:
        self.func = func
        self.size = size
        self.n = n
        self.output_file = Path(output_file)
        self.cache_path = self.output_file.parent / "parts"
        self.worker_files = [self.cache_path / f"{output_file.stem}_part_{i:03d}.pkl" for i in range(n)]

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _worker(self, worker_idx, counter):
        out = {}

        pbar = tqdm(total=self.size, desc="Parallel processing") if worker_idx == 0 else None
        idx_last = 0

        while True:
            with counter.get_lock():
                idx = counter.value

                if idx >= self.size:
                    break

                counter.value += 1

            out[idx] = self.func(idx)

            if pbar is not None:
                pbar.update(idx - idx_last)
                idx_last = idx

        if pbar is not None:
            pbar.close()
        with open(self.worker_files[worker_idx], "wb") as f:
            pickle.dump(out, f)
        print(f"Wrote {self.worker_files[worker_idx]}")

    def aggregate_result(self):
        output_all = [None] * self.size

        for worker_file in tqdm(self.worker_files, desc="Aggregating"):
            with open(worker_file, "rb") as f:
                out = pickle.load(f)
            for k, v in out.items():
                output_all[k] = v

        output_all = [o for o in output_all if o is not None]

        with open(self.output_file, "wb") as f:
            pickle.dump(output_all, f)

        return output_all
            

    def run(self):
        counter = Value("i", 0)

        procs = [Process(target=self._worker, args=(i, counter)) for i in range(self.n)]

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        print("Aggregating...")
        return self.aggregate_result()
    