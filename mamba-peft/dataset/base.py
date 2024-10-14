from abc import abstractmethod, ABC
import random
import transformers
import torch
from pathlib import Path
from tqdm import tqdm
import pickle

from utils.parallel_processor_fs import ParallelProcessorFS


class DatasetBase(ABC):
    shuffle_seeds = [
        123,
        321,
        532,
        523,
    ]
    
    def __init__(self, tokenizer: transformers.AutoTokenizer, path: str, split="train", prompt_prefix=None,
                 use_cache=True, num_parallel_workers=16, subset_size=None, mode="lm", max_seqlen=None):
        super().__init__()

        self.path = path
        self.split = split

        self.sep = "###"
        self.eot = "<|endoftext|>"
        self.tokenizer = tokenizer  
        self.ignore_index = -100
        self.data = None
        self.prompt_prefix = prompt_prefix
        self.prompt_prefix_ids = None
        self.mode = mode
        self.max_seqlen = max_seqlen

        if use_cache:
            cache_file_stem = self.get_cache_name()

            if subset_size is not None:
                cache_file_stem += f"_{subset_size}"

            cache_file = Path("data") / path.replace("/", "_") / f"{cache_file_stem}.pkl"
            if not cache_file.exists():
                if num_parallel_workers > 0:
                    assert subset_size is None
                    data_ind = list(range(len(self)))

                    self.data = ParallelProcessorFS(self.preproc, len(data_ind), num_parallel_workers, cache_file).run()
                else:
                    data_ind = list(range(len(self)))
                    
                    if subset_size is not None:
                        random.Random(0).shuffle(data_ind)
                        data_ind = data_ind[:subset_size]

                    self.data = [self.preproc(idx) for idx in tqdm(data_ind)]
                    self.data = [d for d in self.data if d is not None]

                    if use_cache:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_file, "wb") as f:
                            pickle.dump(self.data, f)
            else:
                with open(cache_file, "rb") as f:
                    self.data = pickle.load(f)

    def get_cache_name(self):
        return f"cache_{self.path.replace('/', ' ')}_{self.split}"

    def encode(self, seq):
        return torch.LongTensor(self.tokenizer.encode(seq))

    def preproc(self, idx):
        input, label = self.get_input_label(idx)
        input_prepoc, label_preproc = self.preproc_input_label(input, label)
        input_ids, label_ids = self.encode(input_prepoc), self.encode(label_preproc)

        if self.max_seqlen is not None and (input_ids.shape[0] + label_ids.shape[0]) > self.max_seqlen:
            return None

        return input_ids, label_ids
    
    def get_ids(self, idx):
        return self.data[idx]

    def __getitem__(self, idx):
        input_ids, label_ids = self.get_ids(idx)

        if self.prompt_prefix is not None:
            if self.prompt_prefix_ids is None:
                self.prompt_prefix_ids = self.encode(self.prompt_prefix)

            input_ids = torch.cat([self.prompt_prefix_ids, input_ids])

        if self.mode == "lm":
            ids = torch.cat([input_ids, label_ids])
            label_len = label_ids.shape[0]

            input_ids = ids[:-1]
            label_ids = torch.nn.functional.pad(ids[-label_len:], (input_ids.shape[0] - label_len, 0), value=self.ignore_index)
        elif self.mode == "gen":
            pass
        else:
            raise Exception(self.mode)
        
        return dict(input_ids=input_ids, label_ids=label_ids)

    @abstractmethod
    def get_input_label(self, idx):
        pass

    @abstractmethod
    def preproc_input_label(self, input, label):
        pass

    @abstractmethod
    def compute_metrics(self, eval_preds):
        pass


class NluDatasetBase(DatasetBase):
    def label_int_to_str(self, label):
        assert 0 <= label <= 9
        return str(label)
    
    def label_str_to_int(self, label):
        return int(label)
    
    def preproc_input_label(self, input, label):
        if isinstance(label, int):
            label = self.label_int_to_str(label)

        return input + self.tokenizer.sep_token, label   # + self.tokenizer.eos_token
    
    # workaround for old cache file, which store input and label concatenated
    def get_ids(self, idx):
        sample = self.data[idx]
    
        if not isinstance(sample, (tuple, list)):
            input_ids, label_ids = sample[:-1], sample[-1:]
        else:
            input_ids, label_ids = sample

        return input_ids, label_ids


class NlgDatasetBase(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preproc_input_label(self, input, label):
        return input + self.tokenizer.sep_token, label + self.tokenizer.eos_token
    