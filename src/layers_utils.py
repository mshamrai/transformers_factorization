import torch.nn as nn
import numpy as np
from src.factor_linear import FactorLinear


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def replace_layer(module, layer, name_path):
    name_prefix = name_path[0]
    name_suffix = name_path[1:]

    for child_name, child in module.named_children():
        if name_prefix == child_name:
            if len(name_suffix) > 1:
                return replace_layer(child, layer, name_suffix)
            elif len(name_suffix) == 1:
                setattr(child, name_suffix[0], layer)
                print("Layer replaced!")
                return
            elif len(name_suffix) == 0:
                setattr(module, name_prefix, layer)
                print("Layer replaced!")
                return
    print(f"Didn't find {name_path}")


def group_layers(layers):
    groups = {}
    for name, layer in layers.items():
        data = layer.weight.detach().numpy()
        shape = data.shape
        min_shape = min(shape)
        bias = None
        if isinstance(layer, nn.Linear):
            bias = layer.bias

        if min_shape in groups:
            groups[min_shape].append(
                {
                    "name": name,
                    "tensor": data,
                    "shape": shape,
                    "transpose": shape[0] < shape[1],
                    "bias": bias,
                }
            )
        else:
            groups[min_shape] = [
                {
                    "name": name,
                    "tensor": data,
                    "shape": shape,
                    "transpose": shape[0] < shape[1],
                    "bias": bias,
                }
            ]

    return groups


def concat_group(group):
    result = []
    for l in group:
        tensor = l["tensor"].T if l["transpose"] else l["tensor"]
        result.append(tensor)
    return np.concatenate(result)


def update_model(model, group, U, V):
    cumulative_n = 0
    V = nn.Parameter(V)
    for l in group:
        n_i = l["shape"][0] if not l["transpose"] else l["shape"][1]
        U_i = U[cumulative_n : cumulative_n + n_i, :]
        cumulative_n += n_i

        U_i = nn.Parameter(U_i)

        fac_linear = FactorLinear(U_i, V, l["bias"], transpose=l["transpose"])
        replace_layer(model, fac_linear, l["name"].split("."))
