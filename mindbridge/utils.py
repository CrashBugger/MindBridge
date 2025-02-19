import copy
import gc
import json
import logging
import random

import torch
from einops import einops
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from torch import nn


class PrefixData:
    def __init__(self, prefix_data_path):
        with open(prefix_data_path, 'r') as f:
            self.data = json.load(f)
        blank_data_ratio = 0.33
        print("blank_data_ratio ", blank_data_ratio, " /", len(self.data))
        self.data += ["" for _ in range(int(len(self.data) * blank_data_ratio))]

    def get_one(self):
        data_item = random.choice(self.data)
        if data_item != "":
            data_item = data_item + " "
        return data_item


def get_data(category, start, end, dataset="zsre", print=print):
    def clean_dataset(dataset):
        new_dataset = []
        for i in dataset:
            if i["alt"] != "":
                new_dataset.append(i)
        print(f"clean dataset: {len(dataset)} -> {len(new_dataset)}")
        return new_dataset

    if dataset == "zsre":
        if category == "Out":
            with open('../data/ZsRE/zsre_mend_train.json', 'r') as f:
                data = json.load(f)
                data = clean_dataset(data)
                print(f"zsre Out-of-scope data, len(data): {len(data)}, start: {start}, end: {end}")
                data = data[start:end]

        elif category == "In":
            with open('../data/ZsRE/zsre_mend_train.json', 'r') as f:
                data = json.load(f)
                data = clean_dataset(data)
                print(f"zsre In scope data, len(data): {len(data)}, start: {start}, end: {end}")
                data = data[start:end]

    elif dataset == "mcf":
        if category == "Out":
            with open('../data/CF/multi_counterfact_slpited_loc.json', 'r') as f:
                data = json.load(f)
                print(f"cf len(data): {len(data)}, start: {start}, end: {end}")
                data = data[start:end]
                data = [j for i in data for j in i["loc_train"]]
        elif category == "In":
            with open('../data/CF/multi_counterfact_slpited_loc.json', 'r') as f:
                data = json.load(f)
                print(f"cf len(data): {len(data)} , start: {start}, end: {end}")
                data = data[start:end]
    return data


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def empty_cache():
    torch.cuda.empty_cache()

    gc.collect()


def add_memory_prefix(target_model_tok, device, target_model_tokenizer):
    target_model_tok = copy.deepcopy(target_model_tok)

    if target_model_tok["attention_mask"] is not None:
        target_model_tok["attention_mask"] = torch.cat(
            [
                torch.full((target_model_tok["attention_mask"].shape[0], 1), 1).to(device),
                target_model_tok["attention_mask"]],
            dim=1)

    target_model_tok["input_ids"] = torch.cat(
        [
            torch.full((target_model_tok["input_ids"].shape[0], 1), target_model_tokenizer.pad_token_id).to(
                device),
            target_model_tok["input_ids"]],
        dim=1)

    return target_model_tok


class CrossModel(torch.nn.Module):
    def __init__(self, memory_model, target_model, mlp, target_model_tokenizer, memory_model_tokenizer):
        super(CrossModel, self).__init__()
        self.memory_model = memory_model
        self.target_model = target_model
        self.mlp = mlp

        self.target_model_tokenizer = target_model_tokenizer
        self.memory_model_tokenizer = memory_model_tokenizer

        if "llama" in type(target_model).__name__.lower():
            self.target_model_embedding = target_model.model.embed_tokens
        else:
            self.target_model_embedding = target_model.transformer.wte

        self.set_memory_model_trainable(False)
        self.set_target_model_trainable(False)
        self.set_projection_model_trainable(False)

        self.is_cut_off_memory_model = False

        self.config = self.target_model.config
        self.device = None

    def _set_model_trainable(self, model, trainable):
        for param in model.parameters():
            param.requires_grad = trainable

    def set_memory_model_trainable(self, trainable):
        self._set_model_trainable(self.memory_model, trainable)

    def cut_off_memory_model(self, cut_off):
        self.is_cut_off_memory_model = cut_off

    def set_target_model_trainable(self, trainable):
        self._set_model_trainable(self.target_model, trainable)

    def set_projection_model_trainable(self, trainable):
        self._set_model_trainable(self.mlp, trainable)

    def get_memory_model_outputs(self, txt):
        device = next(self.memory_model.parameters()).device
        memory_model_tok = self.memory_model_tokenizer(txt, return_tensors="pt").to(device)
        memory_model_outputs = self.memory_model(**memory_model_tok)
        cls_embedding = memory_model_outputs.hidden_states[-1][:, 0, :]
        cls_transformed = self.mlp(cls_embedding)
        return cls_transformed

    def generate(self, input_prompt_ids, max_length, do_sample=False):
        if self.is_cut_off_memory_model:
            return self.target_model.generate(input_prompt_ids, max_length=max_length, do_sample=do_sample)
        else:
            decoded_text = [self.target_model_tokenizer.decode(input_prompt_ids[i], skip_special_tokens=True) for i in
                            range(input_prompt_ids.shape[0])]
            E_m_projected = self.get_memory_model_outputs(decoded_text)

            with torch.no_grad():
                inputs_embeds = self.target_model.get_input_embeddings()(input_prompt_ids)

            inputs_embeds = torch.cat([einops.rearrange(E_m_projected, "b d -> b 1 d"), inputs_embeds],
                                      dim=1).to(next(self.target_model.parameters()).dtype)

            with torch.no_grad():
                generated_tokens = self.target_model.generate(inputs_embeds=inputs_embeds, max_length=max_length)

            return generated_tokens

    def forward(self, **kwargs):
        def decode_and_reencode(tokenize_input, source_tokenizer, target_tokenizer):
            decoded_text = [source_tokenizer.decode(tokenize_input["input_ids"][i], skip_special_tokens=True) for i in
                            range(tokenize_input["input_ids"].shape[0])]

            encoded_result = target_tokenizer(decoded_text, padding=True, return_tensors="pt").to(self.device)

            return encoded_result

        class Output:
            pass

        output = Output()
        output.logits = None

        if not self.is_cut_off_memory_model:
            memory_model_tok = decode_and_reencode(kwargs, self.target_model_tokenizer, self.memory_model_tokenizer)
            memory_model_outputs = self.memory_model(**memory_model_tok)

            cls_embedding = memory_model_outputs.hidden_states[-1][:, 0, :]
            cls_transformed = self.mlp(cls_embedding)

            def hook_wte_forward(module, input, output):

                return torch.cat([einops.rearrange(cls_transformed, "b d -> b 1 d"), output[:, 1:, :]], dim=1)

            hooks = self.target_model_embedding.register_forward_hook(hook_wte_forward)

            target_model_tok = add_memory_prefix(kwargs, device=self.device,
                                                 target_model_tokenizer=self.target_model_tokenizer)

            target_model_outputs = self.target_model(**target_model_tok)

            hooks.remove()
            output.logits = target_model_outputs.logits[:, 1:]
            output.past_key_values = None
        else:
            target_model_outputs = self.target_model(**kwargs)
            output.logits = target_model_outputs.logits
            output.past_key_values = None
        return output

    def train_forward(self, target_model_tok, memory_model_tok, locality_tok, device):

        answer_lengths = locality_tok["target_model_attention_mask"].sum(dim=1)
        last_token_positions = answer_lengths - 1

        with torch.no_grad():
            locality_original_logits = self.target_model(
                input_ids=locality_tok["target_model_input_ids"],
                attention_mask=locality_tok["target_model_attention_mask"]
            ).logits

        original_last_logits = torch.stack([
            locality_original_logits[i, pos, :]
            for i, pos in enumerate(last_token_positions)
        ])

        def cal_LM_loss(lm_logits, labels):
            assert torch.all(torch.isfinite(lm_logits)), "LM loss logits contain NaN or inf."
            assert torch.all(torch.isfinite(labels)), "LM loss labels contain NaN or inf."

            labels = labels.to(lm_logits.device)

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss

        def cal_kl_loss(student_logits, teacher_logits, temperature=1.5):
            assert torch.all(torch.isfinite(student_logits)), "Student logits contain NaN or inf."
            assert torch.all(torch.isfinite(teacher_logits)), "Teacher logits contain NaN or inf."

            teacher_logits = teacher_logits / temperature
            student_logits = student_logits / temperature

            log_student = torch.nn.functional.log_softmax(student_logits, dim=-1)
            teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

            teacher_probs = teacher_probs.clamp(min=1e-8, max=1 - 1e-7)

            return torch.nn.functional.kl_div(
                log_student,
                teacher_probs,
                reduction='batchmean',
                log_target=False
            ) * (temperature ** 2)

        input_ids = torch.cat([memory_model_tok["input_ids"], locality_tok["memory_model_input_ids"]], dim=0)
        attention_mask = torch.cat([memory_model_tok["attention_mask"], locality_tok["memory_model_attention_mask"]],
                                   dim=0)
        memory_model_outputs = self.memory_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = memory_model_outputs.hidden_states[-1][:, 0, :]
        cls_transformed = self.mlp(cls_embedding)

        def hook_wte_forward(module, input, output):
            return torch.cat([einops.rearrange(cls_transformed, "b d -> b 1 d").to(output.dtype), output[:, 1:, :]],
                             dim=1)

        hooks = self.target_model_embedding.register_forward_hook(hook_wte_forward)

        knowledge_end = target_model_tok["input_ids"].shape[0]
        target_model_tok["input_ids"] = torch.cat(
            [target_model_tok["input_ids"], locality_tok["target_model_input_ids"]], dim=0)
        target_model_tok["attention_mask"] = torch.cat(
            [target_model_tok["attention_mask"], locality_tok["target_model_attention_mask"]], dim=0)

        target_model_tok = add_memory_prefix(target_model_tok, device, self.target_model_tokenizer)
        target_model_outputs = self.target_model(input_ids=target_model_tok["input_ids"],
                                                 attention_mask=target_model_tok["attention_mask"])

        loss_mem = cal_LM_loss(target_model_outputs.logits[:knowledge_end, 1:], target_model_tok["labels"])

        target_model_outputs_logits_locality = target_model_outputs.logits[knowledge_end:, 1:]

        student_last_logits = torch.stack([
            target_model_outputs_logits_locality[i, pos, :]
            for i, pos in enumerate(last_token_positions)
        ])
        loss_kl = cal_kl_loss(student_last_logits, original_last_logits)

        hooks.remove()
        return {"loss_mem": loss_mem, "loss_kl": loss_kl}

    def set_test(self, is_test):
        self.test = is_test

    def get_tok_text(self, text, device=None):
        if not device:
            device = next(self.parameters()).device
        target_model_tok = self.target_model_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            device)
        memory_model_tok = self.memory_model_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            device)
        return target_model_tok, memory_model_tok


class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projection, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiCrossModel(CrossModel):
    def __init__(self, memory_models, target_model, mlps, target_model_tokenizer, memory_model_tokenizers):
        """
        Initialize MultiCrossModel with multiple memory models

        Args:
            memory_models (list): List of memory models
            target_model: Target model
            mlps (list): List of MLP projections corresponding to each memory model
            target_model_tokenizer: Tokenizer for target model
            memory_model_tokenizers (list): List of tokenizers for memory models
        """

        super().__init__(memory_models[0], target_model, mlps[0],
                         target_model_tokenizer, memory_model_tokenizers[0])

        self.memory_models = nn.ModuleList(memory_models)
        self.mlps = nn.ModuleList(mlps)
        self.memory_model_tokenizers = memory_model_tokenizers
        self.num_memory_models = len(memory_models)

        self.memory_model = None
        self.mlp = None
        self.memory_model_tokenizer = None

        self.set_memory_models_trainable(False)
        self.set_target_model_trainable(False)
        self.set_projection_models_trainable(False)

    def set_memory_models_trainable(self, trainable):
        """Set trainable status for all memory models"""
        for model in self.memory_models:
            for param in model.parameters():
                param.requires_grad = trainable

    def set_projection_models_trainable(self, trainable):
        """Set trainable status for all projection MLPs"""
        for mlp in self.mlps:
            for param in mlp.parameters():
                param.requires_grad = trainable

    def decode_and_reencode(self, tokenize_input, source_tokenizer, target_tokenizer):
        decoded_text = [source_tokenizer.decode(tokenize_input["input_ids"][i], skip_special_tokens=True) for i in
                        range(tokenize_input["input_ids"].shape[0])]

        encoded_result = target_tokenizer(decoded_text, padding=True, return_tensors="pt").to(self.device)

        return encoded_result

    def add_multiple_memory_prefix(self, target_model_tok, device, target_model_tokenizer):
        """
        Add multiple prefix tokens to the input tensors, one for each memory model

        Args:
            target_model_tok (dict): Input tensors
            device: Target device
            target_model_tokenizer: Tokenizer for target model

        Returns:
            dict: Modified input tensors with multiple prefix tokens
        """
        target_model_tok = copy.deepcopy(target_model_tok)
        num_prefixes = self.num_memory_models

        if target_model_tok["attention_mask"] is not None:
            prefix_attention = torch.full(
                (target_model_tok["attention_mask"].shape[0], num_prefixes),
                1
            ).to(device)
            target_model_tok["attention_mask"] = torch.cat(
                [prefix_attention, target_model_tok["attention_mask"]],
                dim=1
            )

        prefix_input_ids = torch.full(
            (target_model_tok["input_ids"].shape[0], num_prefixes),
            target_model_tokenizer.pad_token_id
        ).to(device)
        target_model_tok["input_ids"] = torch.cat(
            [prefix_input_ids, target_model_tok["input_ids"]],
            dim=1
        )

        return target_model_tok

    def forward(self, **kwargs):
        """
        Forward pass combining multiple memory models' outputs

        Returns:
            Output object with logits and past_key_values
        """

        class Output:
            pass

        output = Output()
        output.logits = None

        if not self.is_cut_off_memory_model:

            cls_transformed_list = []

            for idx, (memory_model, mlp, memory_tokenizer) in enumerate(
                    zip(self.memory_models, self.mlps, self.memory_model_tokenizers)):
                memory_model_tok = self.decode_and_reencode(
                    kwargs,
                    self.target_model_tokenizer,
                    memory_tokenizer
                )

                memory_outputs = memory_model(**memory_model_tok)
                cls_embedding = memory_outputs.hidden_states[-1][:, 0, :]
                cls_transformed = mlp(cls_embedding)
                cls_transformed_list.append(cls_transformed)

            all_cls_transformed = torch.stack(cls_transformed_list, dim=1)

            def hook_wte_forward(module, input, output):

                prefix_embeddings = einops.rearrange(
                    all_cls_transformed,
                    "b n d -> b n d"
                ).to(output.dtype)

                return torch.cat([prefix_embeddings, output[:, self.num_memory_models:, :]], dim=1)

            hooks = self.target_model_embedding.register_forward_hook(hook_wte_forward)

            target_model_tok = self.add_multiple_memory_prefix(
                kwargs,
                device=self.device,
                target_model_tokenizer=self.target_model_tokenizer
            )

            target_outputs = self.target_model(**target_model_tok)

            hooks.remove()

            output.logits = target_outputs.logits[:, self.num_memory_models:]
            output.past_key_values = None

        else:

            target_outputs = self.target_model(**kwargs)
            output.logits = target_outputs.logits
            output.past_key_values = None

        return output


def get_cross_model(memory_model_path, target_model_path, mlp_path=None, memory_model_tokenizer_path=None):
    memory_model = AutoModelForMaskedLM.from_pretrained(memory_model_path, output_hidden_states=True)
    memory_model_tokenizer = AutoTokenizer.from_pretrained(
        memory_model_path if not memory_model_tokenizer_path else memory_model_tokenizer_path)

    gpt_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16)

    gpt_tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    input_dim = memory_model.get_input_embeddings().weight.size(1)
    mlp = Projection(input_dim=input_dim, hidden_dim=input_dim * 2,
                     output_dim=gpt_model.get_input_embeddings().weight.size(1))
    if mlp_path:
        mlp.load_state_dict(torch.load(mlp_path))
    return CrossModel(memory_model, gpt_model, mlp, gpt_tokenizer, memory_model_tokenizer)


def generate_experiment_id(length=20):
    import random
    import string
    import time
    from datetime import datetime

    original_seed = random.getstate()

    random.seed(time.time())

    exp_id = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    random.setstate(original_seed)

    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + exp_id
