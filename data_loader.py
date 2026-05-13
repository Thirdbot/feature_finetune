import argparse
from pathlib import Path

import numpy as np
import segyio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./seismic_data/')
    parser.add_argument('--save_dir', type=str, default='./save_data/')
    return parser.parse_args()

def read_segyio(file_name):
    try:
        with segyio.open(file_name) as f:
            return segyio.tools.cube(f) # strict mode
    except ValueError as exc:
        print(f'  geometry inference failed: {exc}')
        print('  reading traces with strict=False')

    with segyio.open(file_name, strict=False) as f:
        return segyio.tools.collect(f.trace[:]) # non strict mode

def read_dir(dir_name, save_dir=None):
    dir_path = Path(dir_name)
    save_path = Path(save_dir) if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    for el in dir_path.iterdir():
        if el.is_file():
            print(f'read at {el.name}')
            data = read_segyio(el)
            print(f'  shape: {data.shape}')
            if save_path:
                output_file = save_path / f'{el.stem}.npy'
                np.save(output_file, data)
                print(f'  saved to {output_file}')


def list_npy(dir_name):
    dir_path = Path(dir_name)
    return sorted(dir_path.glob('*.npy'))


def normalize_patch(data):
    data = np.nan_to_num(data)
    limit = np.percentile(np.abs(data), 99)
    if limit == 0:
        limit = 1
    data = np.clip(data, -limit, limit) / limit
    return data


def quantize_patch(data, bins=9):
    data = normalize_patch(data)
    data = np.rint(data * bins).astype(np.int16)
    return data


def values_to_text(data):
    rows = []
    for row in data:
        rows.append(' '.join(str(int(el)) for el in row))
    return '\n'.join(rows)


def seismic_tokens(bins=9):
    tokens = ['<seismic_reconstruct>', '<seismic_sep>']
    for el in range(-bins, bins + 1):
        tokens.append(f'<amp_{el}>')
    return tokens


def values_to_token_text(data):
    rows = []
    for row in data:
        rows.append(''.join(f'<amp_{int(el)}>' for el in row))
    return '<seismic_sep>'.join(rows)


class SeismicSelfSupervisedDataset(Dataset):
    def __init__(
        self,
        data_dir='./save_data/',
        samples_per_file=1000,
        patch_traces=16,
        patch_samples=64,
        seed=42,
    ):
        self.files = list_npy(data_dir)
        if not self.files:
            raise FileNotFoundError(f'no .npy files found in {data_dir}')

        self.data = [np.load(el, mmap_mode='r') for el in self.files]
        self.samples_per_file = samples_per_file
        self.patch_traces = patch_traces
        self.patch_samples = patch_samples
        self.seed = seed

    def __len__(self):
        return len(self.files) * self.samples_per_file

    def make_patch(self, data, rng):
        if data.ndim == 2:
            max_trace = max(data.shape[0] - self.patch_traces, 1)
            max_sample = max(data.shape[1] - self.patch_samples, 1)
            trace_start = rng.integers(0, max_trace)
            sample_start = rng.integers(0, max_sample)
            patch = data[
                trace_start:trace_start + self.patch_traces,
                sample_start:sample_start + self.patch_samples,
            ]
            return np.asarray(patch)

        if data.ndim == 3:
            axis0 = rng.integers(0, data.shape[0])
            max_trace = max(data.shape[1] - self.patch_traces, 1)
            max_sample = max(data.shape[2] - self.patch_samples, 1)
            trace_start = rng.integers(0, max_trace)
            sample_start = rng.integers(0, max_sample)
            patch = data[
                axis0,
                trace_start:trace_start + self.patch_traces,
                sample_start:sample_start + self.patch_samples,
            ]
            return np.asarray(patch)

        raise ValueError(f'expected 2D or 3D seismic data, got shape {data.shape}')

    def make_text(self, patch):
        patch = quantize_patch(patch)
        masked = patch.copy()
        mask_row = patch.shape[0] // 2
        target = patch[mask_row:mask_row + 1]
        masked[mask_row, :] = 0

        return (
            '### Instruction:\n'
            'Reconstruct the masked center seismic trace from the quantized neighboring seismic patch.\n\n'
            '### Input:\n'
            f'{values_to_text(masked)}\n\n'
            '### Response:\n'
            f'{values_to_text(target)}'
        )

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        rng = np.random.default_rng(self.seed + idx)
        patch = self.make_patch(self.data[file_idx], rng)
        return {'text': self.make_text(patch)}


class SeismicMultimodalSelfSupervisedDataset(Dataset):
    def __init__(
        self,
        data_dir='./save_data/',
        samples_per_file=1000,
        patch_traces=64,
        patch_samples=64,
        image_size=224,
        target_bins=9,
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
        self.target_bins = target_bins
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

    def make_vision_x(self, patch):
        patch = normalize_patch(patch).astype(np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        patch = F.interpolate(
            patch,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )[0]
        patch = patch.repeat(3, 1, 1)
        return patch.unsqueeze(0).unsqueeze(0)

    def make_text(self, patch):
        patch = quantize_patch(patch, self.target_bins)
        center = patch[patch.shape[0] // 2]
        prompt = '<image><seismic_reconstruct>'
        answer = values_to_token_text(center[None, :])
        return prompt, answer

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        rng = np.random.default_rng(self.seed + idx)
        patch = self.make_patch(self.data[file_idx], rng)
        prompt, answer = self.make_text(patch)
        return {
            'vision_x': self.make_vision_x(patch),
            'prompt': prompt,
            'answer': answer,
        }


class SeismicMultimodalCollator:
    def __init__(self, tokenizer, max_length=512):
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


class SeismicVisualSSLDataset(Dataset):
    def __init__(
        self,
        data_dir='./save_data/',
        samples_per_file=1000,
        patch_traces=64,
        patch_samples=64,
        image_size=224,
        vision_patch_size=14,
        mask_ratio=0.5,
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
        self.vision_patch_size = vision_patch_size
        self.mask_ratio = mask_ratio
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

    def make_image(self, patch):
        patch = normalize_patch(patch).astype(np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        patch = F.interpolate(
            patch,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )[0]
        return patch.repeat(3, 1, 1)

    def make_mask(self, rng):
        grid_size = self.image_size // self.vision_patch_size
        mask = rng.random(grid_size * grid_size) < self.mask_ratio
        if not mask.any():
            mask[rng.integers(0, mask.shape[0])] = True
        return torch.from_numpy(mask)

    def apply_mask(self, image, mask):
        image = image.clone()
        grid_size = self.image_size // self.vision_patch_size
        mask = mask.view(grid_size, grid_size)
        for row in range(grid_size):
            for col in range(grid_size):
                if mask[row, col]:
                    h0 = row * self.vision_patch_size
                    w0 = col * self.vision_patch_size
                    image[
                        :,
                        h0:h0 + self.vision_patch_size,
                        w0:w0 + self.vision_patch_size,
                    ] = 0
        return image

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        rng = np.random.default_rng(self.seed + idx)
        patch = self.make_patch(self.data[file_idx], rng)
        image = self.make_image(patch)
        mask = self.make_mask(rng)
        return {
            'masked_image': self.apply_mask(image, mask),
            'target_image': image,
            'patch_mask': mask,
        }


def main():
    args = parse()
    read_dir(args.data_dir, args.save_dir)

if __name__ == '__main__':
    main()
