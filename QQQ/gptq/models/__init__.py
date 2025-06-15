# from .llama import gptq_llama_func, QuantizedLlamaForCausalLM
# from .qwen2 import gptq_qwen2_func, QuantizedQwen2ForCausalLM
from .gemma3 import gptq_gemma3_func, QuantizedGemma3ForCausalLM

# _GPTQ_MODEL_FUNC = {"llama": gptq_llama_func, "qwen2": gptq_qwen2_func}
_GPTQ_MODEL_FUNC = {"gemma3": gptq_gemma3_func}

_QUANTIZED_MODEL_CLASS = {
    # "llama": QuantizedLlamaForCausalLM,
    # "qwen2": QuantizedQwen2ForCausalLM,
    "gemma3": QuantizedGemma3ForCausalLM,
}


def get_gptq_model_func(model_type):
    if model_type in _GPTQ_MODEL_FUNC:
        return _GPTQ_MODEL_FUNC[model_type]
    else:
        raise NotImplementedError


def get_quantized_model_class(model_type):
    if model_type in _QUANTIZED_MODEL_CLASS:
        return _QUANTIZED_MODEL_CLASS[model_type]
    else:
        raise NotImplementedError
