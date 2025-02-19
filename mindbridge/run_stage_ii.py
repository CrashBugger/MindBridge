import argparse
import os
import pprint
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import yaml
import random
import utils
import torch
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from accelerate import Accelerator


class DatasetOutscope(Dataset):
    def __init__(self, data_locality, target_model_tokenizer, memory_model_tokenizer, dataset_name,
                 add_prefix_data=False, prefix_data=None):
        self.data_locality = data_locality
        self.target_model_tokenizer = target_model_tokenizer
        self.memory_model_tokenizer = memory_model_tokenizer
        self.len_data_locality = len(data_locality)
        self.dataset_name = dataset_name

        self.add_prefix_data = add_prefix_data
        self.prefix_data = prefix_data

    def __len__(self):
        return self.len_data_locality

    def _extract_info(self, example, dataset_name):
        res = None
        if dataset_name == 'zsre':
            res = example["src"]
        elif dataset_name == "mcf":
            res = example
        if self.add_prefix_data:
            res = self.prefix_data.get_one() + res
        return res

    def __getitem__(self, idx):
        example = self.data_locality[idx]
        src_example = self._extract_info(example, self.dataset_name)
        target_model_inputs = self.target_model_tokenizer(src_example, truncation=True, padding=True,
                                                          max_length=max_length,
                                                          return_tensors="pt")
        x = {}
        for i in target_model_inputs:
            x["target_model_" + i] = target_model_inputs[i][0]

        memory_model_inputs = self.memory_model_tokenizer(src_example, truncation=True, padding=True,
                                                          max_length=max_length,
                                                          return_tensors="pt")
        for i in memory_model_inputs:
            x["memory_model_" + i] = memory_model_inputs[i][0]
        return x


class CustomDataset(Dataset):
    def __init__(self, data, batch_size, dataset_name, add_prefix_data=False, prefix_data=None):
        self.data = data
        self.shuffle = True
        self.batch_size = batch_size
        self.indexes = list(range(len(data)))
        self.dataset_name = dataset_name

        self.add_prefix_data = add_prefix_data
        self.prefix_data = prefix_data

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def _extract_info(self, example, dataset_name):
        example_extracts = {}

        return example_extracts

    def __getitem__(self, idx):
        example = self.data[idx]
        example_extracts = {}

        if self.dataset_name == "zsre":
            example_extracts = example
        elif self.dataset_name == "mcf":
            example_extracts["subject"] = example["requested_rewrite"]["subject"]
            example_extracts["alt"] = example["requested_rewrite"]["target_new"]["str"]
            example_extracts["src"] = example["requested_rewrite"]["prompt"].format(
                example["requested_rewrite"]["subject"])

        if self.add_prefix_data:
            example_extracts["src"] = self.prefix_data.get_one() + example_extracts["src"]

        return example_extracts

    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.indexes)

        for i in range(0, len(self.data), self.batch_size):
            batch_indexes = self.indexes[i:i + self.batch_size]
            batch_data = [self[idx] for idx in batch_indexes]
            batch_dict = {}

            batch_dict["src"] = [i["src"] for i in batch_data]
            batch_dict["alt"] = [i["alt"] for i in batch_data]
            yield batch_dict


def get_dict_to_device(a, device):
    return {key: val.to(device) for key, val in a.items()}


def tokenize(tokenizer, prompt_inputs, alt_inputs, device):
    prompts = prompt_inputs
    labels = alt_inputs

    mask_token = -100

    full_prompt = [p + ' ' + l for p, l in zip(prompts, labels)]
    prompt_attention_mask = \
        tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)[
            "attention_mask"]

    num_prompt_toks = [sum(i) for i in prompt_attention_mask]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tokens["labels"] = tokens["input_ids"].clone()

    for i in range(len(num_prompt_toks)):
        tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    tokens = get_dict_to_device(tokens, device)
    return tokens


def collate_fn_datasetoutscope(batch):
    res = {k: [] for k in batch[0].keys()}
    for i in batch:
        for k in i.keys():
            res[k].append(i[k])
    return res


def padding_sequence(memory_model_tok, target_model_tok, locality_tok, memory_model_tok_padding_id,
                     target_model_tok_padding_id):
    max_len = max(
        memory_model_tok['input_ids'].size(1),
        target_model_tok['input_ids'].size(1),
        max(seq.size(0) for seq in locality_tok['target_model_input_ids']),
        max(seq.size(0) for seq in locality_tok['memory_model_input_ids'])
    )

    device = memory_model_tok['input_ids'].device

    if memory_model_tok['input_ids'].size(1) < max_len:
        pad_len = max_len - memory_model_tok['input_ids'].size(1)

        memory_model_tok['input_ids'] = torch.cat([
            memory_model_tok['input_ids'],
            torch.full((memory_model_tok['input_ids'].size(0), pad_len),
                       memory_model_tok_padding_id,
                       dtype=torch.long,
                       device=device)
        ], dim=1)

        memory_model_tok['attention_mask'] = torch.cat([
            memory_model_tok['attention_mask'],
            torch.zeros(memory_model_tok['attention_mask'].size(0), pad_len,
                        dtype=torch.long,
                        device=device)
        ], dim=1)

    if target_model_tok['input_ids'].size(1) < max_len:
        pad_len = max_len - target_model_tok['input_ids'].size(1)

        target_model_tok['input_ids'] = torch.cat([
            target_model_tok['input_ids'],
            torch.full((target_model_tok['input_ids'].size(0), pad_len),
                       target_model_tok_padding_id,
                       dtype=torch.long,
                       device=device)
        ], dim=1)

        target_model_tok['attention_mask'] = torch.cat([
            target_model_tok['attention_mask'],
            torch.zeros(target_model_tok['attention_mask'].size(0), pad_len,
                        dtype=torch.long,
                        device=device)
        ], dim=1)

        if 'labels' in target_model_tok:
            target_model_tok['labels'] = torch.cat([
                target_model_tok['labels'],
                torch.full((target_model_tok['labels'].size(0), pad_len),
                           -100,
                           dtype=torch.long,
                           device=device)
            ], dim=1)

    for key in locality_tok.keys():
        if 'input_ids' in key:
            if 'target_model' in key:
                padding_id = target_model_tok_padding_id
            elif 'memory_model' in key:
                padding_id = memory_model_tok_padding_id
            else:
                raise ValueError("Invalid key in locality_tok: {}".format(key))
            padded_seqs = []
            for seq in locality_tok[key]:
                if seq.size(0) < max_len:
                    padded_seq = torch.cat([
                        seq,
                        torch.full((max_len - len(seq),),
                                   padding_id,
                                   dtype=torch.long,
                                   device=device)
                    ])
                elif seq.size(0) == max_len:
                    padded_seq = seq[:max_len]
                else:
                    raise ValueError("Invalid input_ids size: {}".format(seq.size()))
                padded_seqs.append(padded_seq)
            locality_tok[key] = torch.stack(padded_seqs)

        elif 'attention_mask' in key:
            padded_masks = []
            for mask in locality_tok[key]:
                if mask.size(0) < max_len:
                    padded_mask = torch.cat([
                        mask,
                        torch.zeros(max_len - len(mask),
                                    dtype=torch.long,
                                    device=device)
                    ])
                elif mask.size(0) == max_len:
                    padded_mask = mask[:max_len]
                else:
                    raise ValueError("Invalid attention mask size: {}".format(mask.size()))
                padded_masks.append(padded_mask)
            locality_tok[key] = torch.stack(padded_masks)

    return memory_model_tok, target_model_tok, locality_tok


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to stage ii config file", type=str,
                        default="./config/stage_ii.yaml")
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)
    max_length = args.max_length
    args.exp_id = utils.generate_experiment_id()

    with (open(f"./results/pretrain/{args.memory_exp_id}/config_my.yaml", "r") as f):
        config_dict = yaml.safe_load(f)
        args.memory_model_name = config_dict["memory_model"]
        args.dataset = config_dict["dataset"]
        args.memory_exp_details = config_dict

        args.add_prefix_data = config_dict["add_prefix_data"]
        args.memory_exp_path = "./results/pretrain/{}".format(args.memory_exp_id)
        args.add_prefix_data = False if args.dataset == "zsre" else args.add_prefix_data

        args.in_scope_index_start = config_dict["in_scope_index_start"]
        args.in_scope_index_end = config_dict["in_scope_index_end"]
        args.out_of_scope_index_start = config_dict["out_of_scope_index_start"]
        args.out_of_scope_index_end = config_dict["out_of_scope_index_end"]
    args.save_model_path = f"./results/bridge/{args.exp_id}"
    prefix_data = None
    if args.add_prefix_data:
        prefix_data = utils.PrefixData(args.prefix_data_path)

    log_file = f"./log/bridge_{args.dataset}_{args.target_model.split('/')[1]}.log"

    logger = utils.setup_logger(log_file)
    logger.info("\n" * 5 + f"{'#' * 20} Starting...")

    logger.info("\n" + pprint.pformat(vars(args)))

    model = utils.get_cross_model(args.memory_exp_path, args.target_model,
                                  memory_model_tokenizer_path=args.memory_model_name)

    model.target_model_tokenizer.pad_token = model.target_model_tokenizer.eos_token \
        if not model.target_model_tokenizer.pad_token else model.target_model_tokenizer.pad_token

    if args.dataset == "zsre":
        data_knowledge = utils.get_data("In", args.in_scope_index_start,
                                        args.in_scope_index_end, "zsre", print=logger.info)
        data_locality = utils.get_data("Out", args.out_of_scope_index_start,
                                       args.out_of_scope_index_end, "zsre", print=logger.info)
    elif args.dataset == "mcf":
        data_knowledge = utils.get_data("In", args.in_scope_index_start, args.in_scope_index_end, "mcf",
                                        print=logger.info)
        data_locality = utils.get_data("Out", args.out_of_scope_index_start,
                                       args.out_of_scope_index_end, "mcf", print=logger.info)

    logger.info(f"Len in scope: {len(data_knowledge)}, Out-of-scope: {len(data_locality)}")

    dataset = CustomDataset(data_knowledge, batch_size=args.batch_size,
                            dataset_name=args.dataset,
                            add_prefix_data=args.add_prefix_data,
                            prefix_data=prefix_data)
    dataset_locality = DatasetOutscope(data_locality,
                                       memory_model_tokenizer=model.memory_model_tokenizer,
                                       target_model_tokenizer=model.target_model_tokenizer,
                                       dataset_name=args.dataset,
                                       add_prefix_data=args.add_prefix_data,
                                       prefix_data=prefix_data)

    dataset_locality_loader = torch.utils.data.DataLoader(dataset_locality,
                                                          batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=4,
                                                          collate_fn=collate_fn_datasetoutscope)

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation_steps)
    logger.info(accelerator.state)

    device = accelerator.device
    optimizer = AdamW(model.parameters(), lr=args.lr_rate)

    real_len_dataset = int(len(dataset) / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=3 * real_len_dataset,
                                                   num_training_steps=real_len_dataset * args.num_epochs)

    model, dataset_locality_loader, optimizer, lr_scheduler = accelerator.prepare(model, dataset_locality_loader,
                                                                                  optimizer, lr_scheduler)

    dataset_locality_iter = iter(dataset_locality_loader)
    model.set_projection_model_trainable(True)
    for epoch in range(args.num_epochs):

        if epoch >= args.num_epochs * args.train_projection_ratio:
            model.set_memory_model_trainable(True)

        model.train()

        logger.info(
            f"Epoch: {epoch} " + f"Trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)} " +
            f"Lr : {lr_scheduler.get_last_lr()}")

        loss_dict_epoch = {"loss_kl": 0, "loss_mem": 0, "loss_count": 0}

        progress_bar = tqdm(enumerate(dataset), desc=f"Epoch {epoch}")
        for idx, batch in progress_bar:
            with accelerator.accumulate(model):
                try:
                    locality_tok = next(dataset_locality_iter)
                except:
                    dataset_locality_iter = iter(dataset_locality_loader)
                    locality_tok = next(dataset_locality_iter)

                alt_inputs, prompt_inputs = batch['alt'], batch['src']

                target_model_tok = tokenize(model.target_model_tokenizer, prompt_inputs,
                                            alt_inputs, device)

                memory_model_tok = model.memory_model_tokenizer(prompt_inputs, return_tensors="pt", padding=True,
                                                                truncation=True,
                                                                max_length=max_length)
                memory_model_tok = get_dict_to_device(memory_model_tok, device)

                memory_model_tok, target_model_tok, locality_tok = padding_sequence(memory_model_tok,
                                                                                    target_model_tok,
                                                                                    locality_tok,
                                                                                    memory_model_tok_padding_id=model.memory_model_tokenizer.pad_token_id,
                                                                                    target_model_tok_padding_id=model.target_model_tokenizer.pad_token_id)
                optimizer.zero_grad()

                loss_dict = model.train_forward(memory_model_tok=memory_model_tok, target_model_tok=target_model_tok,
                                                locality_tok=locality_tok, device=device)
                loss = loss_dict["loss_kl"] + loss_dict["loss_mem"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=3.0)

                optimizer.step()
                lr_scheduler.step()

                progress_bar.set_postfix(loss_kl=loss_dict["loss_kl"].item(), loss_mem=loss_dict["loss_mem"].item())
                loss_dict_epoch["loss_kl"] += loss_dict["loss_kl"].item()
                loss_dict_epoch["loss_mem"] += loss_dict["loss_mem"].item()
                loss_dict_epoch["loss_count"] += 1

        logger.info(f"Epoch {epoch}, "
                    f"Loss_kl: {loss_dict_epoch['loss_kl'] / (loss_dict_epoch['loss_count'] + 1)}, "
                    f"Loss_mem: {loss_dict_epoch['loss_mem'] / (loss_dict_epoch['loss_count'] + 1)}")

    os.makedirs(args.save_model_path, exist_ok=True)
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    torch.save(model.mlp.state_dict(), os.path.join(args.save_model_path, "mlp.pt"))
    model.memory_model.save_pretrained(args.save_model_path)

    with open(os.path.join(args.save_model_path, "config_my.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    logger.info(f"Save to {args.save_model_path}")
