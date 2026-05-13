import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from align_train import add_lora, load_model, move_batch, save_model, set_bridge_trainable
from data_loader import list_npy, normalize_patch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vision_path', type=str, default='./adapter/seismic_visual_ssl/vision_encoder')
    parser.add_argument('--vision_code_path', type=str, default='./models/VL3')
    parser.add_argument('--lang_path', type=str, default='./merged_llm/k2')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./save_data/')
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./adapter/seismic_fewshot')

    parser.add_argument('--samples_per_file', type=int, default=1000)
    parser.add_argument('--patch_traces', type=int, default=64)
    parser.add_argument('--patch_samples', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--synthetic_fault_ratio', type=float, default=0.5)
    parser.add_argument('--max_throw', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--max_new_tokens', type=int, default=16)

    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    return parser.parse_args()


def fault_tokens():
    tokens = ['<fault_detect>', '<fault_present>', '<fault_absent>']
    for el in range(0, 91, 15):
        tokens.append(f'<fault_angle_{el}>')
    for el in range(-8, 9):
        tokens.append(f'<fault_throw_{el}>')
    return tokens


def repair_vision_encoder_files(vision_path, vision_code_path):
    vision_path = Path(vision_path)
    vision_code_path = Path(vision_code_path)
    config_file = vision_path / 'config.json'
    if not config_file.exists():
        return

    with config_file.open('r', encoding='utf-8') as file:
        config = json.load(file)

    changed = False
    if 'model_type' not in config:
        config['model_type'] = 'videollama3_vision_encoder'
        changed = True
    if 'auto_map' not in config:
        config['auto_map'] = {
            'AutoConfig': 'configuration_videollama3_encoder.Videollama3VisionEncoderConfig',
            'AutoModel': 'modeling_videollama3_encoder.Videollama3VisionEncoderModel',
        }
        changed = True
    if 'architectures' not in config:
        config['architectures'] = ['Videollama3VisionEncoderModel']
        changed = True

    if changed:
        with config_file.open('w', encoding='utf-8') as file:
            json.dump(config, file, indent=2)
            file.write('\n')

    for file_name in ['configuration_videollama3_encoder.py', 'modeling_videollama3_encoder.py']:
        target_file = vision_path / file_name
        source_file = vision_code_path / file_name
        if not target_file.exists() and source_file.exists():
            shutil.copyfile(source_file, target_file)


def setup_fault_tokenizer(model):
    tokenizer = model.tokenizer
    tokens = fault_tokens()
    tokens.extend(['<|endofchunk|>', '<image>'])
    tokenizer.add_special_tokens({'additional_special_tokens': tokens})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.model.lang_encoder.resize_token_embeddings(len(tokenizer))
    model_dtype = next(model.model.lang_encoder.parameters()).dtype
    model.model.perceiver.to(dtype=model_dtype)
    model.model.lang_encoder.gated_cross_attn_layers.to(dtype=model_dtype)
    return tokenizer


def make_image(patch, image_size):
    patch = normalize_patch(patch).astype(np.float32)
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
    patch = F.interpolate(
        patch,
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False,
    )[0]
    patch = patch.repeat(3, 1, 1)
    return patch.unsqueeze(0).unsqueeze(0)


def inject_fault(patch, rng, max_throw=8):
    patch = patch.copy()
    throw = int(rng.integers(1, max_throw + 1))
    if rng.random() < 0.5:
        throw = -throw

    slope = float(rng.uniform(-0.8, 0.8))
    center_trace = patch.shape[0] / 2
    center_sample = patch.shape[1] / 2

    for trace_idx in range(patch.shape[0]):
        boundary = int(center_sample + slope * (trace_idx - center_trace))
        boundary = max(1, min(boundary, patch.shape[1] - 1))
        patch[trace_idx, boundary:] = np.roll(patch[trace_idx, boundary:], throw)

    angle = int(round(abs(np.degrees(np.arctan(slope))) / 15) * 15)
    angle = max(0, min(angle, 90))
    return patch, angle, throw


class SyntheticFaultFewshotDataset(Dataset):
    def __init__(
        self,
        data_dir='./save_data/',
        samples_per_file=1000,
        patch_traces=64,
        patch_samples=64,
        image_size=224,
        synthetic_fault_ratio=0.5,
        max_throw=8,
        seed=42,
    ):
        self.files = list_npy(data_dir)
        if not self.files:
            raise FileNotFoundError(f'no .npy files found in {data_dir}')

        self.data = [np.load(el, mmap_mode='r') for el in self.files]
        self.samples_per_file = samples_per_file
        self.patch_traces = patch_traces
        self.patch_samples = patch_samples
        self.image_size = image_size
        self.synthetic_fault_ratio = synthetic_fault_ratio
        self.max_throw = max_throw
        self.seed = seed

    def __len__(self):
        return len(self.files) * self.samples_per_file

    def make_patch(self, data, rng):
        if data.ndim == 2:
            max_trace = max(data.shape[0] - self.patch_traces, 1)
            max_sample = max(data.shape[1] - self.patch_samples, 1)
            trace_start = rng.integers(0, max_trace)
            sample_start = rng.integers(0, max_sample)
            return np.asarray(data[
                trace_start:trace_start + self.patch_traces,
                sample_start:sample_start + self.patch_samples,
            ])

        if data.ndim == 3:
            axis0 = rng.integers(0, data.shape[0])
            max_trace = max(data.shape[1] - self.patch_traces, 1)
            max_sample = max(data.shape[2] - self.patch_samples, 1)
            trace_start = rng.integers(0, max_trace)
            sample_start = rng.integers(0, max_sample)
            return np.asarray(data[
                axis0,
                trace_start:trace_start + self.patch_traces,
                sample_start:sample_start + self.patch_samples,
            ])

        raise ValueError(f'expected 2D or 3D seismic data, got shape {data.shape}')

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        rng = np.random.default_rng(self.seed + idx)
        patch = self.make_patch(self.data[file_idx], rng)

        if rng.random() < self.synthetic_fault_ratio:
            patch, angle, throw = inject_fault(patch, rng, self.max_throw)
            answer = f'<fault_present><fault_angle_{angle}><fault_throw_{throw}>'
        else:
            answer = '<fault_absent>'

        return {
            'vision_x': make_image(patch, self.image_size),
            'prompt': '<image><fault_detect>',
            'answer': answer,
        }


class LabeledFaultFewshotDataset(Dataset):
    def __init__(self, label_file, image_size=224):
        self.records = []
        with open(label_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f'no records found in {label_file}')
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        data = np.load(record['file'], mmap_mode='r')
        trace_start = int(record.get('trace_start', 0))
        sample_start = int(record.get('sample_start', 0))
        patch_traces = int(record.get('patch_traces', 64))
        patch_samples = int(record.get('patch_samples', 64))
        patch = np.asarray(data[
            trace_start:trace_start + patch_traces,
            sample_start:sample_start + patch_samples,
        ])

        label = record.get('label', 'absent')
        if label in ['present', 'fault_present', 1, True]:
            angle = int(record.get('angle', 45))
            angle = int(round(angle / 15) * 15)
            angle = max(0, min(angle, 90))
            throw = int(record.get('throw', 0))
            throw = max(-8, min(throw, 8))
            answer = f'<fault_present><fault_angle_{angle}><fault_throw_{throw}>'
        else:
            answer = '<fault_absent>'

        return {
            'vision_x': make_image(patch, self.image_size),
            'prompt': '<image><fault_detect>',
            'answer': answer,
        }


class FaultCollator:
    def __init__(self, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        vision_x = torch.stack([el['vision_x'] for el in batch])
        eos_token = self.tokenizer.eos_token or ''
        texts = [el['prompt'] + el['answer'] + eos_token for el in batch]
        prompts = [el['prompt'] for el in batch]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        labels = tokenized['input_ids'].clone()
        for idx in range(labels.shape[0]):
            prompt_len = int(prompt_tokens['attention_mask'][idx].sum().item())
            labels[idx, :prompt_len] = -100
        labels[tokenized['attention_mask'] == 0] = -100

        return {
            'vision_x': vision_x,
            'lang_x': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
        }


def make_dataset(args):
    if args.label_file:
        return LabeledFaultFewshotDataset(args.label_file, args.image_size)

    return SyntheticFaultFewshotDataset(
        data_dir=args.data_dir,
        samples_per_file=args.samples_per_file,
        patch_traces=args.patch_traces,
        patch_samples=args.patch_samples,
        image_size=args.image_size,
        synthetic_fault_ratio=args.synthetic_fault_ratio,
        max_throw=args.max_throw,
        seed=args.seed,
    )


def prepare_model(args):
    repair_vision_encoder_files(args.vision_path, args.vision_code_path)
    model = load_model(args)
    tokenizer = setup_fault_tokenizer(model)
    model.requires_grad_(False)
    set_bridge_trainable(model)

    if args.use_lora:
        model = add_lora(model, args)
        set_bridge_trainable(model)

    return model, tokenizer


def train(args):
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = prepare_model(args)
    model.to(device)
    model.train()

    dataset = make_dataset(args)
    collator = FaultCollator(tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    params = [param for param in model.parameters() if param.requires_grad]
    print_trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f'fewshot epoch {epoch}')
        for batch_idx, batch in enumerate(progress):
            batch = move_batch(batch, device)
            output = model(**batch)
            loss = output.loss / args.grad_accum
            loss.backward()

            is_accum_step = (batch_idx + 1) % args.grad_accum == 0
            is_last_step = batch_idx + 1 == len(loader)
            if is_accum_step or is_last_step:
                grad_norm = get_grad_norm(params)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.set_postfix(
                    loss=float(loss.item() * args.grad_accum),
                    grad_norm=grad_norm,
                )

                if args.save_steps > 0 and step % args.save_steps == 0:
                    save_model(model, tokenizer, args.output_dir)

                if args.max_steps > 0 and step >= args.max_steps:
                    save_model(model, tokenizer, args.output_dir)
                    return

    save_model(model, tokenizer, args.output_dir)


def print_trainable_parameters(model):
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    print(f'trainable parameters: {trainable} / {total}')


def get_grad_norm(params):
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().float().norm().item())
    return total


def infer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = prepare_model(args)
    model.to(device)
    model.eval()

    dataset = make_dataset(args)
    sample = dataset[0]
    vision_x = sample['vision_x'].unsqueeze(0).to(device)
    prompt = tokenizer(
        sample['prompt'],
        return_tensors='pt',
    )
    lang_x = prompt['input_ids'].to(device)
    attention_mask = prompt['attention_mask'].to(device)

    with torch.no_grad():
        output = model.generate(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=False))


def main():
    args = parse()
    if args.mode == 'train':
        train(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
