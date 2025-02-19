import os
import pprint

import examples_utils as eutils

os.environ["CUDA_VISIBLE_DEVICES"] = f"{eutils.get_best_gpu()}"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import argparse

import torch
from pathlib import Path

import yaml
from accelerate import Accelerator

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from easyeditor import (
    MyBaseEditor,
    summary_metrics,
)


def get_model_tok(exp_id):
    from mindbridge import utils
    model_path = Path("../../mindbridge/results/bridge") / exp_id
    config = yaml.safe_load(open(model_path / "config_my.yaml", "r"))
    pprint.pprint(config)
    memory_model_tokenizer_path = config["memory_model_name"]
    mlp_path = model_path / "mlp.pt"

    target_model_path = config["target_model"]
    cross_model = utils.get_cross_model(memory_model_path=model_path,
                                        memory_model_tokenizer_path=memory_model_tokenizer_path,
                                        target_model_path=target_model_path,
                                        mlp_path=mlp_path)
    cross_model.target_model_tokenizer.pad_token = cross_model.target_model_tokenizer.eos_token \
        if not cross_model.target_model_tokenizer.pad_token else cross_model.target_model_tokenizer.pad_token
    cross_model.eval()
    model = cross_model
    tok = model.target_model_tokenizer
    ds_name = config["dataset"]

    return model, tok, ds_name, target_model_path


if __name__ == "__main__":
    sys.argv = ["python xxx.py", "--exp_id", "20250219-170544-ihjO5KJSsfvBNkbt8SJr"]
    REMOTE_ROOT_URL = "https://memit.baulab.info/data/dsets"
    URL_DICT = {
        "MCF": f"{REMOTE_ROOT_URL}/multi_counterfact.json",
        "CF": f"{REMOTE_ROOT_URL}/counterfact.json",
        "ZsRE": f"{REMOTE_ROOT_URL}/zsre_mend_eval.json"
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--ds_size', default=10, type=str)

    args = parser.parse_args()

    # prepare
    model, tok, ds_name, target_mode_name = get_model_tok(exp_id=args.exp_id)
    args.data_type = {"zsre": "ZsRE", "mcf": "MCF", "cf": "CF"}[ds_name]

    accelerator = Accelerator(mixed_precision="fp16")
    model = accelerator.prepare(model)
    model.device = accelerator.device
    model.name_or_path = model.target_model.name_or_path

    # start
    url = URL_DICT[args.data_type]
    data_dir = Path("../../data/")

    if args.data_type == 'ZsRE':
        zsre_loc = data_dir / "ZsRE" / 'zsre_mend_train.json'
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {url}")
            torch.hub.download_url_to_file(url, zsre_loc)
        with open(zsre_loc, "r") as f:
            raw = json.load(f)
            raw = eutils.clean_dataset(raw)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'CF':
        mcf_loc = data_dir / "CF" / "multi_counterfact_slpited_loc.json"
        if not mcf_loc.exists():
            print(f"{mcf_loc} does not exist. Downloading from {url}")
            torch.hub.download_url_to_file(url, mcf_loc)
        with open(mcf_loc, "r") as f:
            raw = json.load(f)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['requested_rewrite']['prompt'].format(edit_data_['requested_rewrite']['subject']) for
                   edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['paraphrase_prompts'][0] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite']['target_new']['str'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['neighborhood_prompts'] for edit_data_ in edit_data]
        locality_ans = [
            [edit_data_['requested_rewrite']["target_true"]["str"]] * len(edit_data_["neighborhood_prompts"]) for
            edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    editor = MyBaseEditor(model, accelerator.device, tok)

    with accelerator.autocast():
        metrics = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            keep_original_weight=True,
            sequential_edit=False,
            test_generation=False,
        )

    if len(metrics) > 0:
        summary_metrics(metrics)
