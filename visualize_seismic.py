import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./save_data/')
    parser.add_argument('--save_dir', type=str, default='./visualize/data/')
    parser.add_argument('--num_sample', type=int, default=1)
    parser.add_argument('--num_slice', type=int, default=1)
    parser.add_argument('--slice_axis', type=int, default=0)
    parser.add_argument('--slice_idx', type=int, default=None)
    parser.add_argument('--slice_width', type=int, default=256)
    parser.add_argument('--clip_percentile', type=float, default=99.0)
    parser.add_argument('--max_size', type=int, default=1600)
    return parser.parse_args()


def shrink_slice(data, max_size=1600):
    step0 = max(data.shape[0] // max_size, 1)
    step1 = max(data.shape[1] // max_size, 1)
    return np.asarray(data[::step0, ::step1])


def get_slice_idx(size, slice_idx=None, slice_num=0, num_slice=1):
    if slice_idx is not None:
        return slice_idx
    return int((slice_num + 1) * size / (num_slice + 1))


def make_2d_slice(data, slice_axis=0, slice_idx=None, slice_num=0, num_slice=1, slice_width=1200, max_size=1600):
    if slice_axis == 0:
        idx = get_slice_idx(data.shape[0], slice_idx, slice_num, num_slice)
        start = max(idx - slice_width // 2, 0)
        end = min(start + slice_width, data.shape[0])
        return shrink_slice(data[start:end, :], max_size)

    if slice_axis == 1:
        idx = get_slice_idx(data.shape[1], slice_idx, slice_num, num_slice)
        start = max(idx - slice_width // 2, 0)
        end = min(start + slice_width, data.shape[1])
        return shrink_slice(data[:, start:end], max_size)

    raise ValueError('slice_axis must be 0 or 1 for 2D seismic data')


def make_slice(data, slice_axis=0, slice_idx=None, slice_num=0, num_slice=1, slice_width=1200, max_size=1600):
    if data.ndim == 2:
        return make_2d_slice(data, slice_axis, slice_idx, slice_num, num_slice, slice_width, max_size)

    if data.ndim != 3:
        raise ValueError(f'expected 2D or 3D seismic data, got shape {data.shape}')

    if slice_idx is None:
        slice_idx = get_slice_idx(data.shape[slice_axis], None, slice_num, num_slice)

    if slice_axis == 0:
        return shrink_slice(data[slice_idx, :, :], max_size)
    if slice_axis == 1:
        return shrink_slice(data[:, slice_idx, :], max_size)
    if slice_axis == 2:
        return shrink_slice(data[:, :, slice_idx], max_size)

    raise ValueError('slice_axis must be 0, 1, or 2')


def normalize_slice(data, clip_percentile=99.0):
    data = np.nan_to_num(data)
    limit = np.percentile(np.abs(data), clip_percentile)
    if limit == 0:
        limit = 1

    data = np.clip(data, -limit, limit)
    data = (data + limit) / (2 * limit)
    return (data * 255).astype(np.uint8)


def resize_image(image, max_size=1600):
    width, height = image.size
    scale = min(max_size / width, max_size / height, 1)
    if scale == 1:
        return image

    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def save_slice(file_name, save_dir, args, slice_num):
    data = np.load(file_name, mmap_mode='r')
    print(f'read at {file_name.name}')
    print(f'  shape: {data.shape}')

    seismic_slice = make_slice(
        data,
        args.slice_axis,
        args.slice_idx,
        slice_num,
        args.num_slice,
        args.slice_width,
        args.max_size,
    )
    image_data = normalize_slice(seismic_slice, args.clip_percentile)
    image = Image.fromarray(image_data, mode='L')
    image = resize_image(image, args.max_size)

    save_file = save_dir / f'{file_name.stem}_axis{args.slice_axis}_slice{slice_num}.png'
    image.save(save_file)
    print(f'  saved to {save_file}')


def read_dir(dir_name, save_dir, args):
    dir_path = Path(dir_name)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(dir_path.glob('*.npy'))[:args.num_sample]
    if not files:
        raise FileNotFoundError(f'no .npy files found in {dir_path}')

    for el in files:
        for slice_num in range(args.num_slice):
            save_slice(el, save_path, args, slice_num)


def main():
    args = parse()
    read_dir(args.data_dir, args.save_dir, args)


if __name__ == '__main__':
    main()
