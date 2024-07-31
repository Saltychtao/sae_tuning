from sae_lens import SAE
import torch.nn as nn
import torch
import functools
from transformers import AutoModelForCausalLM

class SaeHookModule(nn.Module):
    def __init__(self,sae):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(sae.cfg.d_sae),requires_grad=True)
        self.sae = sae


    def forward(self,x):
        if self.sae.normalize_activations == "layer_norm":
            mu = x.mean(dim=-1, keepdim=True)
            x = x - mu
            std = x.std(dim=-1, keepdim=True)
            self.sae.ln_mu = mu
            self.sae.ln_std = std

        return self.sae(self.delta) + x
    

def find_hook_point(model, sae) -> nn.Module:
    hook_name = sae.cfg.hook_name
    splitted = hook_name.split(".")
    layer_index = sae.cfg.hook_layer
    if "resid_post" in splitted[2]:
        return model.h[layer_index]

def add_hook_to_module(module, hook_module):
    if hasattr(module,"_hf_hook") and hasattr(module,"_old_forward"):
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module._hf_hook = hook_module

    def new_forward(module,*args,**kwargs):
        if module._hf_hook.no_grad:
            with torch.no_grad():
                output = module._old_forward(*args,**kwargs)
        else:
            output = module._old_forward(*args,**kwargs)
        return module._hf_hook.forward(output)
    
    module.forward = functools.update_wrapper(functools.partial(new_forward,module),old_forward)
    return module

def get_sae_tuning_model(model, saes):
    for sae in saes:
        hook_point = find_hook_point(model.transformer,sae)
        sae_hook_module = SaeHookModule(sae)
        add_hook_to_module(hook_point,sae_hook_module)

    for name,param in model.named_parameters():
        if "sae" not in name:
            param.requires_grad = False
    return model


model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
saes = [SAE.from_pretrained(release="gpt2-small-resid-post-v5-32k",sae_id="blocks.{}.hook_resid_post".format(layer),device="cpu")[0] for layer in range(4)]

sae_model = get_sae_tuning_model(model,saes)
