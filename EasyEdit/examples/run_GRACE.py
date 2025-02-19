import os
import pprint
import examples_utils as eutils

os.environ["CUDA_VISIBLE_DEVICES"] = f"{eutils.get_best_gpu(n=1)}"
import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from easyeditor import (
    AlphaEditHyperParams,
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)

REMOTE_ROOT_URL = "https://memit.baulab.info/data/dsets"
URL_DICT = {
    "MCF": f"{REMOTE_ROOT_URL}/multi_counterfact.json",
    "CF": f"{REMOTE_ROOT_URL}/counterfact.json",
    "ZsRE": f"{REMOTE_ROOT_URL}/zsre_mend_eval.json"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['ZsRE', 'MCF', 'CF'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', type=int, required=True)
    parser.add_argument('--sequential_edit', action="store_true")
    # parser.add_argument('--num_edits', type=int, default=-1)

    args = parser.parse_args()
    pprint.pprint(args)

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    else:
        raise NotImplementedError

    url = URL_DICT[args.data_type]
    data_dir = Path(args.data_dir)

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
    elif args.data_type == 'MCF':
        # mcf_loc = data_dir / "CF" / 'multi_counterfact.json'
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

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
    )

    print("See results at: ", output_file)
    # kwargs = dict(
    #     pre_file=f"/data/Bugger/Memory/EasyEdit-main/outputs/{args.editing_method}_{args.ds_size}_pre_file.json")

    # hparams.batch_size = 1
    hparams.model_parallel = True if "," in os.environ["CUDA_VISIBLE_DEVICES"] else False
    hparams.fp16 = False
    pprint.pprint(hparams)

    editor = BaseEditor.from_hparams(hparams)

    # 加载pre edit
    pre_file = os.path.join(os.path.dirname(__file__), "pre_file",
                            f"GRACE_{hparams.model_name.replace('/', '_')}_{args.data_type}_{10 * 1000}.json")
    pre_edit = None
    if os.path.exists(pre_file):
        print(f"Loading pre-edit from {pre_file}")
        pre_edit = json.load(open(pre_file, 'r'))
        if args.ds_size > len(pre_edit):
            raise Exception(f"Pre-edit size {len(pre_edit)} is smaller than ds_size {args.ds_size}, using pre-edit.")
        pre_edit = pre_edit[:args.ds_size]
    else:
        print(f"Pre-edit not found at {pre_file}.")

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        keep_original_weight=True,
        sequential_edit=True,
        test_generation=False,
        **dict(pre_edit=pre_edit, pre_file=pre_file)
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)
