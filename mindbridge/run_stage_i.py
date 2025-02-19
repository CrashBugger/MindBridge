import argparse
import copy
import os

import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.chdir(os.path.dirname(__file__))
import yaml

import einops
import collections
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import utils


class TrainWrap(torch.nn.Module):
    def __init__(self, memory_net, device, classify_task=False, memory_provide_task=False,
                 memory_provide_target_model=None,
                 memory_provide_target_tokenizer=None,
                 memory_loss_weight=1,
                 classify_task_weight=1,
                 memory_provide_task_weight=1, ):
        super().__init__()
        self.memory_net = memory_net
        self._set_trainable(self.memory_net, True)

        self.device = device

        if classify_task:
            self.classify_task_head = torch.nn.Sequential(
                torch.nn.Linear(memory_net.config.hidden_size, memory_net.config.hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(memory_net.config.hidden_size // 2, 2)
            )

        if memory_provide_task:
            self.target_dummy_model = memory_provide_target_model

            self.memory_provide_target_tokenizer = memory_provide_target_tokenizer

            self.memory_projection = utils.Projection(input_dim=memory_net.config.hidden_size,
                                                      hidden_dim=memory_net.config.hidden_size * 2,
                                                      output_dim=memory_provide_target_model.config.hidden_size)

        self.memory_loss_weight = memory_loss_weight
        self.classify_task_weight = classify_task_weight
        self.memory_provide_task_weight = memory_provide_task_weight

        if "distil" in self.target_dummy_model.__class__.__name__.lower():
            self.target_dummy_model_wordembed = self.target_dummy_model.distilbert.embeddings.word_embeddings
        else:
            self.target_dummy_model_wordembed = self.target_dummy_model.bert.embeddings.word_embeddings

        self.train_all_except_target_model()

    def _set_trainable(self, model, trainable):
        for i in model.parameters():
            i.requires_grad = trainable

    def train_only_provide_and_classify(self):
        self._set_trainable(self.memory_net, False)
        self._set_trainable(self.target_dummy_model, False)

        self._set_trainable(self.classify_task_head, True)
        self._set_trainable(self.memory_projection, True)

    def train_all_except_target_model(self):
        self._set_trainable(self.memory_net, True)
        if hasattr(self, 'classify_task_head'):
            self._set_trainable(self.classify_task_head, True)
        if hasattr(self, 'memory_projection'):
            self._set_trainable(self.memory_projection, True)
        if hasattr(self, "target_dummy_model"):
            self._set_trainable(self.target_dummy_model, False)

    @torch.no_grad()
    def classify_eval(self, classify_task_input):
        input_ids = classify_task_input["input_ids"]
        attention_mask = classify_task_input["attention_mask"]
        outputs = self.memory_net(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1][:, 0, :]
        classify_logits = self.classify_task_head(hidden_states)
        return classify_logits

    @torch.no_grad()
    def memory_provide_eval(self, memory_provide_input):
        input_ids = memory_provide_input["src_input_ids"]
        attention_mask = memory_provide_input["src_attention_mask"]
        outputs = self.memory_net(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states[-1][:, 0]  # 获取 [CLS] token 的 hidden state
        cls_transformed = self.memory_projection(hidden_states)

        memory_provide_input = utils.add_memory_prefix(
            target_model_tokenizer=self.memory_provide_target_tokenizer,
            target_model_tok=memory_provide_input,
            device=self.device,
        )

        def hook_wte_forward(module, input, output):
            # 替换嵌入的第一个位置为 transformed hidden states
            return torch.cat(
                [einops.rearrange(cls_transformed, "b d -> b 1 d").to(output.dtype), output[:, 1:, :]],
                dim=1
            )

        hook = self.target_dummy_model_wordembed.register_forward_hook(hook_wte_forward)

        logits = self.target_dummy_model(
            input_ids=memory_provide_input["input_ids"],
            attention_mask=memory_provide_input["attention_mask"]
        ).logits[:, 1:]

        hook.remove()

        return logits

    def forward(self, mask_task_inputs, classify_task_input=None, memory_provide_input=None):

        def concat(a, b):
            if a is not None:
                return torch.cat([a, b], dim=0)
            else:
                return b

        input_ids = None
        attention_mask = None
        if mask_task_inputs:
            input_ids = concat(input_ids, mask_task_inputs["input_ids"])
            attention_mask = concat(attention_mask, mask_task_inputs["attention_mask"])
            mask_task_inputs_start_end = (
                input_ids.shape[0] - mask_task_inputs["input_ids"].shape[0], input_ids.shape[0])

        if classify_task_input:
            input_ids = concat(input_ids, classify_task_input["input_ids"])
            attention_mask = concat(attention_mask, classify_task_input["attention_mask"])
            classify_task_input_start_end = (
                input_ids.shape[0] - classify_task_input["input_ids"].shape[0], input_ids.shape[0])

        if memory_provide_input:
            input_ids = concat(input_ids, memory_provide_input["src_input_ids"])
            attention_mask = concat(attention_mask, memory_provide_input["src_attention_mask"])
            memory_task_input_start_end = (
                input_ids.shape[0] - memory_provide_input["src_input_ids"].shape[0], input_ids.shape[0])

        outputs = self.memory_net(input_ids=input_ids, attention_mask=attention_mask)

        loss = 0

        if mask_task_inputs:
            start, end = mask_task_inputs_start_end
            logits = outputs.logits[start:end]
            mask_loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]),
                                                    mask_task_inputs["labels"].view(-1))
            loss += mask_loss * self.memory_loss_weight

        if classify_task_input:
            start, end = classify_task_input_start_end
            hidden_states = outputs.hidden_states[-1][start:end, 0]
            classify_logits = self.classify_task_head(hidden_states)
            classify_loss = torch.nn.CrossEntropyLoss()(classify_logits.view(-1, classify_logits.shape[-1]),
                                                        classify_task_input["classify_task_labels"].view(-1))
            loss += classify_loss * self.classify_task_weight

        if memory_provide_input:
            def cal_mlm_loss(logits, labels):
                loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                return masked_lm_loss

            start, end = memory_task_input_start_end
            hidden_states = outputs.hidden_states[-1][start:end, 0]
            cls_transformed = self.memory_projection(hidden_states)
            # 添加占位符
            memory_provide_input = utils.add_memory_prefix(target_model_tokenizer=self.memory_provide_target_tokenizer,
                                                           target_model_tok=memory_provide_input, device=self.device, )

            def hook_wte_forward(module, input, output):
                return torch.cat([einops.rearrange(cls_transformed, "b d -> b 1 d").to(output.dtype), output[:, 1:, :]],
                                 dim=1)

            hook = self.target_dummy_model_wordembed.register_forward_hook(hook_wte_forward)
            logits = self.target_dummy_model(input_ids=memory_provide_input["input_ids"],
                                             attention_mask=memory_provide_input["attention_mask"]).logits[:, 1:]
            mlm_loss = cal_mlm_loss(logits, memory_provide_input["labels"])
            hook.remove()

            loss += mlm_loss * self.memory_provide_task_weight

        return loss


class MemoryProvideDataset(Dataset):
    def __init__(self, data, tokenizer, dataset_name, add_prefix_data, prefix_data=None):
        self.data = data
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        self.add_prefix_data = add_prefix_data
        self.prefix_data = prefix_data

    def _extract_info(self, example, dataset_name):
        example_extracts = {}
        if dataset_name == "zsre":
            example_extracts["src"] = example["src"]
            example_extracts["subject"] = example["subject"]
            example_extracts["alt"] = example["alt"]
        elif dataset_name == "mcf":
            example_extracts["subject"] = example["requested_rewrite"]["subject"]
            example_extracts["alt"] = example["requested_rewrite"]["target_new"]["str"]
            example_extracts["src"] = example["requested_rewrite"]["prompt"].format(
                example["requested_rewrite"]["subject"])

        if self.add_prefix_data:
            example_extracts["src"] = self.prefix_data.get_one() + example_extracts["src"]
        return example_extracts

    def mask_data(self, example):
        inputs = self.tokenizer(example['src'] + " " + example["alt"], truncation=True, padding='max_length',
                                max_length=max_length,
                                return_tensors="pt")
        alt_tokens = self.tokenizer(" " + example['alt'], add_special_tokens=False).input_ids

        tokens = inputs['input_ids'][0].clone()
        word_ids = inputs.word_ids()

        labels = -100 * torch.ones((len(tokens),)).to(torch.long)

        new_labels_ids = set()

        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        def mask_tokens(tokens, token_ids):
            for i in range(len(tokens) - len(token_ids) + 1):
                if tokens[i:i + len(token_ids)].tolist() == token_ids:
                    for j in range(len(token_ids)):
                        tokens[i + j] = self.tokenizer.mask_token_id
                        new_labels_ids.add(i + j)
                    break
            return tokens

        tokens = mask_tokens(tokens, alt_tokens)

        if len(new_labels_ids) > 0:
            new_labels_ids = list(new_labels_ids)
            labels[new_labels_ids] = inputs['input_ids'][0][new_labels_ids].to(torch.long)

        inputs['input_ids'] = tokens
        inputs["attention_mask"] = inputs['attention_mask'][0]
        inputs['labels'] = labels
        if 'token_type_ids' in inputs:
            inputs["token_type_ids"] = inputs['token_type_ids'][0]

        src_tok_result = self.tokenizer(example['src'], truncation=True, padding='max_length', max_length=max_length,
                                        return_tensors="pt")
        inputs["src_input_ids"] = src_tok_result["input_ids"][0]
        inputs["src_attention_mask"] = src_tok_result["attention_mask"][0]
        return inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        example = self._extract_info(example, self.dataset_name)
        inputs = self.mask_data(example)
        return inputs


class MaskDataset(Dataset):
    def __init__(self, data, tokenizer, dataset_name, main_token_prob=0.5, other_token_prob=0.1,
                 add_prefix_data=False, prefix_data=None):
        self.data = data
        self.tokenizer = tokenizer
        self.main_token_prob = main_token_prob
        self.other_token_prob = other_token_prob
        self.is_train = True
        self.dataset_name = dataset_name

        self.add_prefix_data = add_prefix_data
        self.prefix_data = prefix_data

    def mask_data(self, example):
        inputs = self.tokenizer(example['src'] + " " + example["alt"] + ".", truncation=True, padding='max_length',
                                max_length=max_length,
                                return_tensors="pt")
        subject_tokens = self.tokenizer(" " + example['subject'], add_special_tokens=False).input_ids
        alt_tokens = self.tokenizer(" " + example['alt'], add_special_tokens=False).input_ids

        tokens = inputs['input_ids'][0].clone()
        word_ids = inputs.word_ids()

        labels = -100 * torch.ones((len(tokens),)).to(torch.long)

        mask_subject = True if random.random() > self.main_token_prob else False
        mask_alt = not mask_subject
        new_labels_ids = set()

        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        def mask_tokens(tokens, token_ids):
            for i in range(len(tokens) - len(token_ids) + 1):
                if tokens[i:i + len(token_ids)].tolist() == token_ids:
                    for j in range(len(token_ids)):
                        tokens[i + j] = self.tokenizer.mask_token_id
                        new_labels_ids.add(i + j)
                    break
            return tokens

        if mask_subject:
            tokens = mask_tokens(tokens, subject_tokens)
        if mask_alt:
            tokens = mask_tokens(tokens, alt_tokens)

        if self.is_train:
            mask = np.random.binomial(1, self.other_token_prob, (len(mapping),))
        else:
            mask = np.zeros(len(mapping))

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels_ids.add(idx)
                tokens[idx] = self.tokenizer.mask_token_id

        if len(new_labels_ids) > 0:
            new_labels_ids = list(new_labels_ids)
            labels[new_labels_ids] = inputs['input_ids'][0][new_labels_ids].to(torch.long)

        inputs['input_ids'] = tokens
        inputs["attention_mask"] = inputs['attention_mask'][0]
        inputs['labels'] = labels
        if 'token_type_ids' in inputs:
            inputs["token_type_ids"] = inputs['token_type_ids'][0]
        return inputs

    def __len__(self):
        return len(self.data)

    def _extract_info(self, example, dataset_name):
        example_extracts = {}
        if dataset_name == "zsre":
            example_extracts["subject"] = example["subject"]
            example_extracts["alt"] = example["alt"]
            example_extracts["src"] = example["src"]
        elif dataset_name == "mcf":
            example_extracts["subject"] = example["requested_rewrite"]["subject"]
            example_extracts["alt"] = example["requested_rewrite"]["target_new"]["str"]
            example_extracts["src"] = example["requested_rewrite"]["prompt"].format(
                example["requested_rewrite"]["subject"])

        if self.add_prefix_data:
            example_extracts["src"] = self.prefix_data.get_one() + example_extracts["src"]
        return example_extracts

    def __getitem__(self, idx):
        example = self.data[idx]
        example = self._extract_info(example, self.dataset_name)
        inputs = self.mask_data(example)
        return inputs


class DatasetKnowledgeAndLocality(Dataset):
    def __init__(self, data_knowledge, data_locality, tokenizer, dataset_name, add_prefix_data,
                 prefix_data=None):
        self.data_locality = data_locality
        self.data_knowledge = data_knowledge
        self.tokenizer = tokenizer
        self.len_data_locality = len(data_locality)
        self.len_data_knowledge = len(data_knowledge)
        self.dataset_name = dataset_name

        self.add_prefix_data = add_prefix_data
        self.prefix_data = prefix_data

    def __len__(self):
        return self.len_data_locality * 2

    def __getitem__(self, idx):
        loc = False
        if idx < len(self.data_locality):
            example = self.data_locality[idx]
            label = 0
            loc = True
        elif idx >= len(self.data_locality):
            idx = idx - len(self.data_locality)
            idx = int(idx * (self.len_data_knowledge / self.len_data_locality))
            example = self.data_knowledge[idx]
            label = 1
        example = self._extract_info(example, self.dataset_name, loc)
        inputs = self.tokenizer(example['src'], truncation=True, padding='max_length', max_length=max_length,
                                return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i][0]
        inputs["classify_task_labels"] = label
        return inputs

    def _extract_info(self, example, dataset_name, loc):
        example_extracts = {}
        if dataset_name == "zsre":
            example_extracts["src"] = example["src"]
        elif dataset_name == "mcf":
            if loc:
                example_extracts["src"] = example
            else:
                example_extracts["src"] = example["requested_rewrite"]["prompt"].format(
                    example["requested_rewrite"]["subject"])

        if self.add_prefix_data:
            example_extracts["src"] = self.prefix_data.get_one() + example_extracts["src"]
        return example_extracts


def calculate_accuracy(preds, labels):
    correct = 0
    total = 0
    for p, l in zip(preds, labels):
        if l != -100:
            total += 1
            if p == l:
                correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_mask(model, dataloader, mask_token_id, dataset_knowledge):
    dataset_knowledge.is_train = False
    model.eval()

    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating")
    for batch in progress_bar:
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch['labels']

            pred_tokens = torch.argmax(logits, dim=-1)

            mask_positions = batch['input_ids'] == mask_token_id
            pred_masked_tokens = pred_tokens[mask_positions]
            true_masked_tokens = labels[mask_positions]

            all_preds.extend(pred_masked_tokens.cpu().numpy())
            all_labels.extend(true_masked_tokens.cpu().numpy())

    accuracy = calculate_accuracy(all_preds, all_labels)
    logger.info(f"Knowledge MASK Accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_memory_provide_mask(model, dataloader, mask_token_id, ):
    model.eval()

    all_preds = []
    all_labels = []

    # 遍历训练集
    progress_bar = tqdm(dataloader, desc="Evaluating")
    for batch in progress_bar:
        with torch.no_grad():
            logits = model.memory_provide_eval(batch)
            labels = batch['labels']

            pred_tokens = torch.argmax(logits, dim=-1)

            mask_positions = batch['input_ids'] == mask_token_id
            pred_masked_tokens = pred_tokens[mask_positions]
            true_masked_tokens = labels[mask_positions]

            all_preds.extend(pred_masked_tokens.cpu().numpy())
            all_labels.extend(true_masked_tokens.cpu().numpy())

    accuracy = calculate_accuracy(all_preds, all_labels)
    logger.info(f"Memory provide MASK Accuracy: {accuracy:.4f}")
    return accuracy


def eval_classify(dataset_locality_dataloader, model, logger):
    model.eval()

    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataset_locality_dataloader, desc="Evaluating Classification Task")
    for batch in progress_bar:
        with torch.no_grad():
            classify_logits = model.classify_eval(batch)

            pred_labels = torch.argmax(classify_logits, dim=-1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(batch["classify_task_labels"].cpu().numpy())

    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0

    logger.info(f"Final Classification Accuracy: {accuracy:.4f}")
    return accuracy


def train(dataset_knowledge_dataloader, dataset_locality_dataloader, dataset_knowledge, model, accelerator, logger,
          classify_and_memory_train=False, dataset_knowledge_provide_dataloader=None):
    model.train()
    dataset_knowledge.is_train = True

    progress_bar = tqdm(dataset_knowledge_dataloader, desc=f"Epoch {epoch + 1}")
    epoch_loss_sum = 0
    dataset_locality_iter = iter(dataset_locality_dataloader)
    dataset_knowledge_provide_iter = iter(dataset_knowledge_provide_dataloader)
    for batch in progress_bar:
        # outputs = model(**batch)
        classify_task_input = None
        memory_provide_input = None
        if classify_and_memory_train:
            try:
                classify_task_input = next(dataset_locality_iter)
            except StopIteration:
                dataset_locality_iter = iter(dataset_locality_dataloader)
                classify_task_input = next(dataset_locality_iter)

            try:
                memory_provide_input = next(dataset_knowledge_provide_iter)
            except:
                dataset_knowledge_provide_iter = iter(dataset_knowledge_provide_dataloader)
                memory_provide_input = next(dataset_knowledge_provide_iter)

        loss = model(mask_task_inputs=batch, classify_task_input=classify_task_input,
                     memory_provide_input=memory_provide_input)

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix(loss=loss.item())
        epoch_loss_sum += loss.item()
    return epoch_loss_sum / len(dataset_knowledge_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to stage i config file", type=str,
                        default="./config/stage_i.yaml")
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)
    args.num_epochs = args.all_epoch + args.mid_epoch + args.inject_epoch
    max_length = args.max_length
    args.exp_id = utils.generate_experiment_id()
    logger = utils.setup_logger(f"./log/stage_i_{args.dataset}.log")
    logger.info("\n" * 5 + "-" * 10 + "Start")
    logger.info("\n" + pprint.pformat(vars(args)))

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

    logger.info(f"num_edit_knowledge:{len(data_knowledge)}, num_edit_locality:{len(data_locality)}")
    prefix_data = None
    if args.add_prefix_data:
        prefix_data = utils.PrefixData(args.prefix_data_path)

    model = AutoModelForMaskedLM.from_pretrained(args.memory_model, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.memory_model)

    dataset_knowledge = MaskDataset(data_knowledge, tokenizer, dataset_name=args.dataset,
                                    prefix_data=prefix_data, add_prefix_data=args.add_prefix_data)
    dataset_knowledge.is_train = True
    dataset_knowledge_provide = MemoryProvideDataset(data_knowledge, tokenizer,
                                                     dataset_name=args.dataset, prefix_data=prefix_data,
                                                     add_prefix_data=args.add_prefix_data)
    dataset_locality = DatasetKnowledgeAndLocality(data_knowledge, data_locality, tokenizer,
                                                   dataset_name=args.dataset, prefix_data=prefix_data,
                                                   add_prefix_data=args.add_prefix_data)

    dataset_knowledge_dataloader = DataLoader(dataset_knowledge, batch_size=args.batch_size, shuffle=True,
                                              num_workers=4)
    dataset_locality_dataloader = DataLoader(dataset_locality, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataset_knowledge_provide_dataloader = DataLoader(dataset_knowledge_provide, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)

    accelerator = Accelerator(mixed_precision="fp16")
    model = TrainWrap(memory_net=model,
                      classify_task=True,

                      memory_provide_task=True,
                      memory_provide_target_model=copy.deepcopy(model),
                      memory_provide_target_tokenizer=copy.deepcopy(tokenizer),

                      memory_loss_weight=args.memory_loss_weight,
                      classify_task_weight=args.classify_task_weight,
                      memory_provide_task_weight=args.memory_provide_task_weight,
                      device=accelerator.device)

    logger.info(f"Trainable parameters (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}", )
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(dataset_knowledge_dataloader) * 5,
                                                   num_training_steps=len(dataset_knowledge_dataloader) * (
                                                       args.num_epochs))
    model, optimizer, dataset_knowledge_dataloader, dataset_locality_dataloader, dataset_knowledge_provide_dataloader = accelerator.prepare(
        model,
        optimizer,
        dataset_knowledge_dataloader,
        dataset_locality_dataloader, dataset_knowledge_provide_dataloader)
    classify_and_memory_train = False

    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        if epoch >= args.inject_epoch:
            classify_and_memory_train = True
            msg = "inject+classify"
            if epoch <= args.inject_epoch + args.mid_epoch:
                model.train_only_provide_and_classify()
                msg = "train_only_provide_and_classify"
            else:
                model.train_all_except_target_model()
                msg = "all"
        else:
            msg = "inject"
        logger.info(f"Epoch {epoch + 1} " + msg)

        if (epoch + 1) % 3 == 0:
            with accelerator.autocast():
                eval_classify(dataset_locality_dataloader, model, logger)
                evaluate_mask(model.memory_net, dataset_knowledge_dataloader, tokenizer.mask_token_id,
                              dataset_knowledge)
                evaluate_memory_provide_mask(model, dataset_knowledge_provide_dataloader, tokenizer.mask_token_id, )

        loss = train(dataset_knowledge_dataloader, dataset_locality_dataloader,
                     dataset_knowledge, model, accelerator, logger, classify_and_memory_train=classify_and_memory_train,
                     dataset_knowledge_provide_dataloader=dataset_knowledge_provide_dataloader)
        logger.info(f"Epoch {epoch + 1} Loss: {loss} " + f"lr: {lr_scheduler.get_last_lr()}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    save_dir = f'./results/pretrain/{args.exp_id}'
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving model to {save_dir}")
    unwrapped_model.memory_net.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "config_my.yaml"), 'w', encoding='utf-8') as file:
        yaml.dump(vars(args), file, allow_unicode=True)
