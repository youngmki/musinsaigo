import os
from typing import Any
import torch
from safetensors.torch import load_file, save_file


LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_PREFIX_UNET = "lora_unet"


def bin_to_safetensors(bin_path: str, safetensors_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_location = torch.device(device)
    bin_state_dict = torch.load(bin_path, map_location=map_location)
    safetensors_state_dict = {}

    for key_bin in bin_state_dict:
        key_safetensors = convert_name_to_safetensors(key_bin)
        safetensors_state_dict[key_safetensors] = bin_state_dict[key_bin]

    save_file(safetensors_state_dict, safetensors_path)


def convert_lora_safetensor_to_diffusers(
    model: Any, model_dir: str, cross_attention_scale: float = 0.5
) -> Any:
    # load lora weight
    model_path = os.path.join(model_dir, "pytorch_lora_weights.safetensors")
    state_dict = load_file(model_path)

    visited = []
    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        # as we have set the alpha beforehand, so just skip

        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            )
            curr_layer = model.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = model.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                else:
                    break
            except AttributeError:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            curr_layer.weight.data += cross_attention_scale * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += cross_attention_scale * torch.mm(
                weight_up, weight_down
            )

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return model


def convert_name_to_safetensors(name: str) -> str:
    parts = name.split(".")

    for i in range(len(parts)):
        if parts[i].isdigit():
            parts[i] = "_" + parts[i]
        if "to" in parts[i] and "lora" in parts[i]:
            parts[i] = parts[i].replace("_lora", ".lora")

    new_parts = []
    for i in range(len(parts)):
        if i == 0:
            new_parts.append(LORA_PREFIX_UNET + "_" + parts[i])
        elif i == len(parts) - 2:
            new_parts.append(parts[i] + "_to_" + parts[i + 1])
            new_parts[-1] = new_parts[-1].replace("_to_weight", "")
        elif i == len(parts) - 1:
            new_parts[-1] += "." + parts[i]
        elif parts[i] != "processor":
            new_parts.append(parts[i])

    new_name = "_".join(new_parts)
    new_name = new_name.replace("__", "_")
    new_name = new_name.replace("_to_out.", "_to_out_0.")
    return new_name
