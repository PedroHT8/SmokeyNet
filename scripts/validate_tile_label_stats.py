"""Validate SmokeyNet precomputed tile labels against a training split."""

import argparse
import pickle
from pathlib import Path


def normalize_image_name(value):
    path = str(value).strip().replace('\\', '/')
    parts = path.split('/')
    if len(parts) < 2:
        raise ValueError(f'Expected fire/image path, got: {value!r}')
    image = Path(parts[-1]).stem
    if image.endswith('_lbl') or image.endswith('_img'):
        image = image[:-4]
    return f'{parts[-2]}/{image}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', required=True, type=Path)
    parser.add_argument('--train-split', required=True, type=Path)
    parser.add_argument('--threshold', default=250, type=int)
    parser.add_argument('--expected-tiles', default=45, type=int)
    args = parser.parse_args()

    with args.stats.open('rb') as stats_file:
        stats = pickle.load(stats_file)
    train_images = [
        normalize_image_name(line)
        for line in args.train_split.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]

    invalid = {name: len(counts) for name, counts in stats.items() if len(counts) != args.expected_tiles}
    if invalid:
        sample = list(invalid.items())[:5]
        raise SystemExit(f'ERROR: entries with invalid tile count: {sample}')

    positive_images = [name for name in train_images if '+' in name]
    negative_images = [name for name in train_images if '+' not in name]
    covered_positive = [name for name in positive_images if name in stats]
    missing_positive = [name for name in positive_images if name not in stats]
    positive_tiles = sum(
        sum(int(pixel_count > args.threshold) for pixel_count in stats[name])
        for name in covered_positive
    )
    total_tiles = (len(covered_positive) + len(negative_images)) * args.expected_tiles
    negative_tiles = total_tiles - positive_tiles

    print(f'stats entries: {len(stats)}')
    print(f'train images: {len(train_images)}')
    print(f'covered positives: {len(covered_positive)}/{len(positive_images)}')
    print(f'negative images retained: {len(negative_images)}')
    print(f'positive images omitted: {len(missing_positive)}')
    print(f'effective train images: {len(covered_positive) + len(negative_images)}')
    print(f'positive tiles: {positive_tiles}')
    print(f'negative tiles: {negative_tiles}')
    print(f'negative/positive tile ratio: {negative_tiles / positive_tiles:.2f}')
    if missing_positive:
        print('first omitted positives:')
        for image_name in missing_positive[:10]:
            print(f'- {image_name}')


if __name__ == '__main__':
    main()
