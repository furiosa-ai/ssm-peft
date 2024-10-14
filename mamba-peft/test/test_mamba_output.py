
import pickle
import sys
from pathlib import Path

from mamba_ssm_peft import get_mamba_peft_model

from mamba_ssm_peft.models.mixer_seq_simple import MambaLMHeadModel
from utils.debug_utils import enable_deterministic

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

import torch




class TestMambaRec(unittest.TestCase):
    peft_cfgs = [
        "cfg/peft/lora/r8/lora_all_no_embed_conv.json",
        "cfg/peft/layer_freeze/A_log_B_C_D_delta_conv.json",

        "cfg/peft/ssm_peft/init_state_conv_state_zero.json",
        "cfg/peft/ssm_peft/init_state_zero.json",

        "cfg/peft/state_offset_tuning/state_offset_tuning.json",
        "cfg/peft/state_offset_tuning/state_offset_tuning_lr.json",
        "cfg/peft/state_offset_tuning/output_tuning.json",

        "cfg/peft/param_transform/BC_bias.json",
        "cfg/peft/param_transform/BC_bias_lr.json",
        "cfg/peft/param_transform/C_bias.json",

        "cfg/peft/bitfit/conv_dt.json"
    ]

    @classmethod
    def setUpClass(cls) -> None:
        # cls.is_save = False

        cls.device = "cuda"
        cls.dtype = torch.float32

        cls.b = 1
        cls.test_in_len = 5
        cls.test_out_len = 3

        cls.input_ids = torch.randint(1, 100, (cls.b, cls.test_in_len), generator=torch.Generator(device=cls.device).manual_seed(0), device=cls.device).long()

    @classmethod
    def assert_equal(cls, name, actual, expected, msg, rtol, atol):
        file = Path("test/data") / (name + ".pkl")

        if cls.is_save:
            assert not file.is_file()

            with open(file, "wb") as f:
                pickle.dump((actual, expected), f)

            return True
        else:
            with open(file, "rb") as f:
                (actual_save, expected_save) = pickle.load(f)

            torch.testing.assert_close(actual, actual_save, rtol=rtol, atol=atol, msg=msg)
            torch.testing.assert_close(expected, expected_save, rtol=rtol, atol=atol, msg=msg)

    @classmethod
    def assert_logits_close(cls, name, actual, expected, msg=None):
        return cls.assert_equal(name + "_logits", actual, expected, msg, rtol=5e-4, atol=5e-4)
    
    @classmethod
    def assert_ids_equal(cls, name, actual, expected, msg=None):
        return cls.assert_equal(name + "_ids", actual, expected, msg, rtol=0, atol=0)

    @classmethod
    def init_uniform(cls, param, a_min, a_max=None):
        v = torch.rand(param.data.shape, dtype=param.data.dtype, device=param.data.device, generator=torch.Generator(device=cls.device).manual_seed(0))

        if a_max is None:
            a_min, a_max = -a_min, a_min

        v = v  * (a_max - a_min) + a_min
        param.data[:] = v

    @classmethod
    def init_param(cls, name, param):
        if "actscale_param" in name:
            cls.init_uniform(param, 0.5, 1.5)
        else:
            cls.init_uniform(param, 0.1)

    @classmethod
    def random_init_trainable(cls, model):
        trainable_params = {
            name: param for name, param in model.named_parameters() if param.requires_grad
        }

        assert len(trainable_params) > 0

        for name, param in trainable_params.items():
            cls.init_param(name, param)

    @classmethod
    def generate(cls, model):
        out_seq = model.generate(
            input_ids=cls.input_ids,
            max_length=cls.test_in_len + cls.test_out_len,
            top_k=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        return out_seq

    @classmethod
    def load_mamba(cls, peft, base=False):
        enable_deterministic()

        model_kwargs = dict(
            dtype=cls.dtype, 
            device=cls.device,
        )

        model = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-130m", 
            **model_kwargs
        )

        if peft is not None:
            if isinstance(peft, list):
                for peft_inst in peft:
                    model = get_mamba_peft_model(model, peft_inst, return_peft_cfg=False)
            else:
                model = get_mamba_peft_model(model, peft, return_peft_cfg=False)
            cls.random_init_trainable(model)

        model.eval()
        return model
    
    @classmethod
    @torch.no_grad()
    def assert_rec_par_output_equal(cls, name, model):
        out_seq_rec = cls.generate(model)

        input_par_ids = out_seq_rec.sequences[:, :-1]
        cls.assert_ids_equal(name, out_seq_rec.sequences[:, :cls.test_in_len], cls.input_ids)
        target_par_ids = out_seq_rec.sequences[:, cls.test_in_len:]
        target_par_logits = torch.stack(out_seq_rec.scores, 1)

        # discard logits for input ids
        pred_par_logits = model(input_par_ids).logits[:, cls.test_in_len-1:]
        pred_par_ids = pred_par_logits.argmax(2)
        
        # self.assert_equal(pred_par_ids, target_par_ids, "Output ids not equal")
        cls.assert_logits_close(name, pred_par_logits, target_par_logits)  # , "Output logits not equal"

    @classmethod
    def template_test_equal(cls, name, peft_cfg):
        def _f(cls):
            cls.assert_rec_par_output_equal(name, cls.load_mamba(peft_cfg, base=False))
        return _f
    
    @classmethod
    def generate_test_functions(cls, is_save) -> None:
        cls.is_save = is_save

        for peft_cfg in cls.peft_cfgs:
            test_name = peft_cfg

            if test_name is not None and isinstance(test_name, (tuple, list)):
                test_name = "_".join([t[len("cfg/peft/"):].split(".")[0].replace("/", "_") for t in test_name])
            else:
                test_name = test_name[len("cfg/peft/"):].split(".")[0].replace("/", "_") if test_name is not None else "no_peft"

            attr_name = f"test_{test_name}_equal"
            assert not hasattr(cls, attr_name)
            setattr(cls, attr_name, cls.template_test_equal(attr_name, peft_cfg))


if __name__ == '__main__':
    TestMambaRec.generate_test_functions(is_save=False)
    unittest.main()
