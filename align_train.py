import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM

from data_loader import (
    SeismicMultimodalCollator,
    SeismicMultimodalSelfSupervisedDataset,
    SeismicVisualSSLDataset,
    seismic_tokens,
)
from register import (
    FlamingoConfig,
    FlamingoModel,
    VideoLLaMA3VisualAdapter,
    clone_shared_tensors,
    register_model,
)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='visual_ssl', choices=['visual_ssl', 'multimodal'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vision_path', type=str, default='./models/VL3')
    parser.add_argument('--lang_path', type=str, default='./merged_llm/k2')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./save_data/')
    parser.add_argument('--output_dir', type=str, default='./adapter/seismic_visual_ssl')

    parser.add_argument('--samples_per_file', type=int, default=1000)
    parser.add_argument('--patch_traces', type=int, default=64)
    parser.add_argument('--patch_samples', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--target_bins', type=int, default=9)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--save_steps', type=int, default=100)

    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--freeze_vision_encoder', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    return parser.parse_args()


class SeismicVisualSSLModel(nn.Module):
    def __init__(self, vision_path, freeze_vision_encoder=False):
        super().__init__()
        self.vision_path = vision_path
        encoder = AutoModel.from_pretrained(vision_path, trust_remote_code=True)
        self.vision_encoder = VideoLLaMA3VisualAdapter(encoder)
        self.patch_size = self.vision_encoder.config.patch_size
        hidden_size = self.vision_encoder.config.hidden_size
        patch_dim = 3 * self.patch_size * self.patch_size
        self.decoder = nn.Linear(hidden_size, patch_dim)

        if freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)

    def forward(self, masked_image, target_image, patch_mask):
        height = (masked_image.shape[-2] // self.patch_size) * self.patch_size
        width = (masked_image.shape[-1] // self.patch_size) * self.patch_size
        masked_image = masked_image[..., :height, :width]
        target_image = target_image[..., :height, :width]

        _, tokens = self.vision_encoder.visual(masked_image)
        tokens = tokens.to(dtype=self.decoder.weight.dtype)
        pred = self.decoder(tokens)

        target = F.unfold(
            target_image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        target = target.transpose(1, 2).to(dtype=pred.dtype)

        patch_mask = patch_mask.to(device=pred.device, dtype=torch.bool)
        return F.mse_loss(pred[patch_mask], target[patch_mask])


def load_model(args):
    register_model()
    if args.model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        return model

    config = FlamingoConfig(
        vision_path=args.vision_path,
        lang_path=args.lang_path,
        n_layers=args.n_layers,
    )
    return FlamingoModel(config)


def freeze_for_bridge_training(model):
    model.requires_grad_(False)
    set_bridge_trainable(model)


def set_bridge_trainable(model):
    for name, param in model.named_parameters():
        if 'perceiver' in name or 'gated_cross_attn_layers' in name:
            param.requires_grad = True


def setup_tokenizer(model, args):
    tokenizer = model.tokenizer
    special_tokens = seismic_tokens(args.target_bins)
    special_tokens.extend(['<|endofchunk|>', '<image>'])
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    model.model.lang_encoder.resize_token_embeddings(len(tokenizer))
    model_dtype = next(model.model.lang_encoder.parameters()).dtype
    model.model.perceiver.to(dtype=model_dtype)
    model.model.lang_encoder.gated_cross_attn_layers.to(dtype=model_dtype)
    return tokenizer


def add_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        task_type='CAUSAL_LM',
    )
    model.model.lang_encoder = get_peft_model(model.model.lang_encoder, lora_config)
    return model


def make_loader(args, tokenizer):
    dataset = SeismicMultimodalSelfSupervisedDataset(
        data_dir=args.data_dir,
        samples_per_file=args.samples_per_file,
        patch_traces=args.patch_traces,
        patch_samples=args.patch_samples,
        image_size=args.image_size,
        target_bins=args.target_bins,
        seed=args.seed,
    )
    collator = SeismicMultimodalCollator(tokenizer, args.max_length)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )


def make_visual_ssl_loader(args, patch_size):
    dataset = SeismicVisualSSLDataset(
        data_dir=args.data_dir,
        samples_per_file=args.samples_per_file,
        patch_traces=args.patch_traces,
        patch_samples=args.patch_samples,
        image_size=args.image_size,
        vision_patch_size=patch_size,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def save_visual_ssl_model(model, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vision_output_path = output_path / 'vision_encoder'
    model.vision_encoder.encoder.save_pretrained(vision_output_path)
    save_portable_vision_encoder_files(model.vision_encoder.encoder, model.vision_path, vision_output_path)
    torch.save(
        {
            'decoder': model.decoder.state_dict(),
            'patch_size': model.patch_size,
            'hidden_size': model.vision_encoder.config.hidden_size,
        },
        output_path / 'visual_ssl_head.pt',
    )


def save_portable_vision_encoder_files(encoder, source_path, vision_output_path):
    config_file = vision_output_path / 'config.json'
    with config_file.open('r', encoding='utf-8') as file:
        config = json.load(file)

    config['model_type'] = 'videollama3_vision_encoder'
    config['architectures'] = ['Videollama3VisionEncoderModel']
    config['auto_map'] = {
        'AutoConfig': 'configuration_videollama3_encoder.Videollama3VisionEncoderConfig',
        'AutoModel': 'modeling_videollama3_encoder.Videollama3VisionEncoderModel',
    }

    with config_file.open('w', encoding='utf-8') as file:
        json.dump(config, file, indent=2)
        file.write('\n')

    source_dir = Path(source_path)
    for file_name in ['configuration_videollama3_encoder.py', 'modeling_videollama3_encoder.py']:
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copyfile(source_file, vision_output_path / file_name)


def save_model(model, tokenizer, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state_dict = clone_shared_tensors(model.state_dict())
    model.save_pretrained(
        output_path,
        safe_serialization=False,
        state_dict=state_dict,
    )
    tokenizer.save_pretrained(output_path)


def train_visual_ssl(args):
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SeismicVisualSSLModel(
        vision_path=args.vision_path,
        freeze_vision_encoder=args.freeze_vision_encoder,
    )
    model.to(device)
    model.train()

    loader = make_visual_ssl_loader(args, model.patch_size)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f'visual ssl epoch {epoch}')
        for batch_idx, batch in enumerate(progress):
            batch = move_batch(batch, device)
            loss = model(**batch) / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.set_postfix(loss=float(loss.item() * args.grad_accum))

                if args.save_steps > 0 and step % args.save_steps == 0:
                    save_visual_ssl_model(model, args.output_dir)

                if args.max_steps > 0 and step >= args.max_steps:
                    save_visual_ssl_model(model, args.output_dir)
                    return

    save_visual_ssl_model(model, args.output_dir)


def train_multimodal(args):
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args)
    tokenizer = setup_tokenizer(model, args)

    freeze_for_bridge_training(model)
    if args.use_lora:
        model = add_lora(model, args)
        set_bridge_trainable(model)

    model.to(device)
    model.train()

    loader = make_loader(args, tokenizer)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f'epoch {epoch}')
        for batch_idx, batch in enumerate(progress):
            batch = move_batch(batch, device)
            output = model(**batch)
            loss = output.loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.set_postfix(loss=float(loss.item() * args.grad_accum))

                if args.save_steps > 0 and step % args.save_steps == 0:
                    save_model(model, tokenizer, args.output_dir)

                if args.max_steps > 0 and step >= args.max_steps:
                    save_model(model, tokenizer, args.output_dir)
                    return

    save_model(model, tokenizer, args.output_dir)


def train(args):
    if args.mode == 'visual_ssl':
        train_visual_ssl(args)
    else:
        train_multimodal(args)


def main():
    args = parse()
    train(args)


if __name__ == '__main__':
    main()
