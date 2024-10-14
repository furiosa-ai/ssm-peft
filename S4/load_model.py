def load_model(
    model,
):
    if model == 's4':
        from load_models.load_s4 import DeepS4D
        return DeepS4D
    elif model == 's6':
        from load_models.load_s6 import DeepS6
        return DeepS6
    elif model == 'mamba':
        from load_models.load_mamba import MixerModel
        return MixerModel
    else:
        raise NotImplementedError(f"Model {model} not implemented.")