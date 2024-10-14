
import json
import os
import shutil

from mamba_ssm_peft.peft import MambaPeftType
from mamba_ssm_peft.peft.sd_lora import SdLoraModel
os.environ["WANDB_PROJECT"] = "mamba-peft"

from pathlib import Path
import torch
import argparse
import numpy as np

import yaml
from mamba_ssm_peft import get_mamba_peft_model, get_trainable_parameters_ratio, load_mamba, load_tokenizer, print_trainable_parameter_names

from mamba_ssm_peft.utils.decoder import create_decoder
from dataset import load_dataset
from trainer.mamba_trainer import MambaTrainer, MambaTrainingArguments


def _lock_share(name):
    path = Path("share/lock") / name
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "x"):
            pass
        return True
    except OSError:
        print(path, "exists")
        return False


def run_train(
    output_dir,
    cfg_path, 
    model,
    data, 
    val_data=None,
    val_data_split="val", 
    tokenizer="EleutherAI/gpt-neox-20b",
    num_epochs=10, 
    prec="bf16", 
    peft=None, 
    optim="adamw_torch", 
    learning_rate=5e-4, 
    gradient_accumulation_steps=1, 
    num_data_workers=8, 
    batch_size=4, 
    eval_gen=None, 
    backend="cuda", 
    debug=False, 
    resume=False, 
    overwrite=False, 
    lock=False, 
    no_save=False, 
    skip_eval=False, 
    eval_epochs=1, 
    min_eval_metric_after_epoch=None,
    seed=42,
    is_sdlora=False):
    
    if overwrite and is_sdlora:
        assert Path(output_dir).exists()

    cfg = {**locals()}

    if not overwrite:
        if lock and _lock_share(output_dir):
            return

        if (Path(output_dir) / "cfg.yaml").exists():
            if resume:
                resume_from_checkpoint = True
            else:
                assert False, str(Path(output_dir) / "cfg.yaml") + " exists!"
        else:
            resume_from_checkpoint = False
    else:
        # assert Path(output_dir).exists()
        resume_from_checkpoint = False

    if not (data.startswith("glue_") or data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000")  or not (no_save and num_epochs > 1)):
        print("Training for more than one epoch without saving ckpts!")

    is_custom_tokenizer = tokenizer != "EleutherAI/gpt-neox-20b"
    tokenizer = load_tokenizer(tokenizer)

    model_kwargs = dict(
        dtype={"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}[prec], 
        device="cuda",
        backend=backend,
    )

    model = load_mamba(
        model, 
        **model_kwargs
    )["model"]

    if peft is not None:
        model, peft_cfg = get_mamba_peft_model(model, peft, return_peft_cfg=True, train_embedding=is_custom_tokenizer, no_print=True)
        assert (is_sdlora and isinstance(model.base_model, SdLoraModel)) or (not is_sdlora and not isinstance(model.base_model, SdLoraModel))
    else:
        peft_cfg = None

    print_trainable_parameter_names(model)

    print("Loaded model")

    train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.dump(cfg, f)

    if eval_gen is not None:
        eval_generator = create_decoder(tokenizer, **eval_gen)
    else:
        eval_generator = None
    
    val_data_module = load_dataset(
        val_data if val_data is not None else data, 
        tokenizer, 
        val_data_split, 
        mode="lm" if eval_gen is None else "gen",
        return_module=True)

    compute_metrics = val_data_module.dataset.compute_metrics

    if debug:
        train_data_module.dataset = torch.utils.data.Subset(train_data_module.dataset, range(8))
        val_data_module.dataset = torch.utils.data.Subset(val_data_module.dataset, range(2))
        num_epochs = 1

    its_per_epoch = int(np.ceil(len(train_data_module.dataset) / batch_size))
    logging_steps = min(50, its_per_epoch)

    os.environ["WANDB_NAME"] = str(output_dir).replace("weights/", "")

    print("Dropping last batch")
    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=MambaTrainingArguments(
            learning_rate=learning_rate,
            max_steps=int(num_epochs * its_per_epoch),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            output_dir=output_dir,
            logging_steps=logging_steps,
            dataloader_num_workers=num_data_workers,
            dataloader_prefetch_factor=2,
            eval_accumulation_steps=128,
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                "cfg_path": cfg_path
            },
            save_strategy="steps" if not no_save else "no",
            evaluation_strategy="steps" if not skip_eval else "no",
            save_steps=int(eval_epochs * its_per_epoch),
            eval_steps=int(eval_epochs * its_per_epoch),
            dataloader_drop_last=True,
            report_to="wandb",
            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,
        eval_generator=eval_generator,
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def get_output_path_for_cfg(cfg_path):
    output_dir = str(Path(cfg_path).parent / Path(cfg_path).stem)
    output_dir = output_dir.replace("cfg/exps/", "")
    output_dir = Path("weights", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = get_output_path_for_cfg(args.cfg)

    train_args = {**cfg, **{k: v for k, v in vars(args).items() if v is not None}, "output_dir": str(output_dir)}
    train_args["cfg_path"] = train_args.pop("cfg")
    if "device" in train_args:
        del train_args["device"]

    is_sdlora = False
    if train_args["peft"] is not None:
        with open(train_args["peft"], "r") as f:
            peft_cfg = json.load(f)

        if peft_cfg["peft_type"] == MambaPeftType.SD_LORA:
            is_sdlora = True

    if is_sdlora:
        # assert not train_args["overwrite"], f"Cannot override SDLora checkpoint"
        if train_args["overwrite"]:
            if Path(train_args["output_dir"]).exists():
                shutil.rmtree(train_args["output_dir"])

        del train_args["overwrite"]
        run_train(**train_args, is_sdlora=True)  # warmup
        run_train(**train_args, is_sdlora=True, overwrite=True)  # training
    else:
        run_train(**train_args)


if __name__ == "__main__":
    main()
