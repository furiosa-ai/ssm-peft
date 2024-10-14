# Mamba SSM PEFT

- Setup
    - Install dependencies
    ```bash
    # Create env
    conda create -n mamba-ssm python=3.10
    conda activate mamba-ssm

    # Install pytorch, e.g.,
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

    # Install mamba
    pip install "causal-conv1d>=1.2.0"
    cd src/mamba
    pip install -e .
    cd -

    # Install requirements
    pip install -r requirements.txt
    ``` 
    - For Spider, download [Spider](https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&authuser=1) and extract to data/xlangai_spider/spider

- Train
    <details open>
    <summary>SDLoRA</summary>

    ```bash
    # spider
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/spider/*channels_and_states*.yaml

    # samsum
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/samsum/*channels_and_states*.yaml

    # dart
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/dart/*channels_and_states*.yaml

    # glue
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/glue/*/*channels_and_states*.yaml
    ```
    </details>

    <details>
    <summary>LoRA (for SDLoRA comparison)</summary>

    ```bash
    # spider
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/spider/*lora_outproj*.yaml

    # samsum
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/samsum/*lora_outproj*.yaml

    # dart
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/dart/*lora_outproj*.yaml

    # glue
    python run_all.py train.py --device 0 --cfg cfg/exps/sdlora/glue/*/*lora_outproj*.yaml
    ```
    </details>

    <details>
    <summary>PEFT Benchmark</summary>

    ```bash
    # spider
    python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/spider/*.yaml

    # spider (mamba-2.8b)
    python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/spider28b/*.yaml

    # samsum
    python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/samsum/*.yaml

    # dart
    python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/dart/*.yaml

    # glue
    python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/glue/*/*.yaml

    # cifar
    python run_all.py train.py --device 0 --cfg cfg/exps/benchmark/cifar/*.yaml
    ```
    </details>

## References

The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).

The official implementation is here: https://github.com/state-spaces/mamba/tree/main
