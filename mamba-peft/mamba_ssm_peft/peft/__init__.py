
import enum
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
        

class MambaPeftType(str, enum.Enum):
    LAYER_FREEZE = "LAYER_FREEZE"
    SEQ_PREFIX = "SEQ_PREFIX"
    TENSOR_LORA = "TENSOR_LORA"
    STATE_OFFSET_TUNING = "STATE_OFFSET_TUNING"
    BITFIT = "BITFIT"
    SSMPEFT = "SSMPEFT"
    PARAM_TRANSFORM = "PARAM_TRANSFORM"
    SD_LORA = "SD_LORA"


def register_peft_tuner(name):
    def _wrap(cls):
        PEFT_TYPE_TO_MODEL_MAPPING[name] = cls
        return cls
    
    return _wrap


def register_peft_config(name):
    def _wrap(cls):
        PEFT_TYPE_TO_CONFIG_MAPPING[name] = cls
        return cls
    
    return _wrap


def _init():
    from .layer_freeze import LayerFreezeModel
    from .state_offset_tuning import StateOffsetTuningModel
    from .bitfit import BitFitModel
    from .ssm_peft import SsmPeftModel
    from .param_transform import ParamTransformModel
    from .sd_lora import SdLoraModel

_init()
