import torch


def export_bert_onnx(model, tokenizer, device, output_path):
    text = "Dummy input."

    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    torch.onnx.export(model, 
                  args=(encoded_input["input_ids"], encoded_input["attention_mask"]), 
                  f=output_path, 
                  export_params=True, 
                  input_names=["input_ids", "attention_mask"],
                  output_names=["last_hidden_state"],
                  dynamic_axes={"input_ids": [0, 1], "attention_mask": [0, 1], "last_hidden_state":[0, 1]})
    

def export_checkpoint(model, output_path):
    torch.save(model, output_path)