import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

import layers_utils
import export


def factorize(grouped_tensor, k):
    U, S, V = np.linalg.svd(grouped_tensor, full_matrices=False)
    U, S, V = U[:,:k], S[:k], V[:k,:]
    V = np.diag(S) @ V
    return torch.tensor(U), torch.tensor(V)


def main(model_name, export_format, k, output_path):
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    groups = layers_utils.group_layers(layers_utils.find_layers(model))
    for group in list(groups.items()):
        group = group[1]
        grouped_tensor = layers_utils.concat_group(group)
        U, V = factorize(grouped_tensor, k)
        layers_utils.update_model(model, group, U, V)

    
    if export_format == "onnx" and "bert" in model_name:
        export.export_bert_onnx(model, tokenizer, device, output_path)
    elif export_format == "checkpoint":
        export.export_checkpoint(model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model factorization script")
    parser.add_argument("--model_name", type=str, help="Name of the model to load")
    parser.add_argument("--output_path", type=str, help="Model output path")
    parser.add_argument("--export_format", type=str, choices=["onnx", "checkpoint"], help="Format to export the model")
    parser.add_argument("-k", type=int, help="A factorization constant")

    args = parser.parse_args()

    main(args.model_name, args.export_format, args.k, args.output_path)
