import os
import torch
from .plot_utils import plot_heatmap
import numpy as np

def view_attention(
    model=None,  # the model object
    model_id=None,
    tokenizer=None,
    prompt=None, 
    save_attention_scores=False,
    save_attention_scores_path=None,
    load_attention_scores_path=None,
    save_fig_path=None,
    plot_figs_per_head=False,
    ignore_first_token=False,
    num_figs_per_row=4,
    start_layer=0,
    end_layer=-1,
    per_token_focus=False,
    numerical_norm='log'
):
    if load_attention_scores_path:
        with open(load_attention_scores_path, 'rb') as f:
            saved_data = torch.load(f)
            attention_scores = saved_data['attention_scores']
            tokens_list = saved_data['tokens_list']
    else:
        assert model is not None and model_id is not None and prompt is not None and tokenizer is not None, \
            "All necessary parameters must be specified!"
        inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)
        tokens_list = list(map(lambda x: x.replace(r'▁', '').replace('Ġ', ''), tokenizer.convert_ids_to_tokens(inputs[0].cpu())))
        with torch.no_grad():
            attention_scores = model(inputs, output_attentions=True)['attentions'][start_layer:end_layer]
        attention_scores = [layer.detach().cpu() for layer in attention_scores]
        
        if save_attention_scores:
            assert save_attention_scores_path is not None, "Specify `save_attention_scores_path` to save attention scores!"
            save_path = save_attention_scores_path
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f'{model_id}_attn_scores.pt'), 'wb') as f:
                saved_data = {'attention_scores': attention_scores, 'tokens_list': tokens_list}
                torch.save(saved_data, f)

    print('Plotting heatmap for attention scores ...')
    plot_heatmap(attention_scores, model_id, plot_figs_per_head, save_fig_path, tokens_list, ignore_first_token, num_figs_per_row, start_layer, per_token_focus, numerical_norm)
