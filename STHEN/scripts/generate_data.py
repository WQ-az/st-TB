
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from simplified_data_generation import SpatiotemporalBifurcationGenerator
import numpy as np


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    n = len(dataset)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return [dataset[i] for i in train_idx], \
           [dataset[i] for i in val_idx], \
           [dataset[i] for i in test_idx]


def main():
    parser = argparse.ArgumentParser(description='Generate bifurcation training data')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                        help='Output directory')
    parser.add_argument('--samples_per_class', type=int, default=8000,
                        help='Number of samples per bifurcation type')
    parser.add_argument('--n_cells', type=int, default=100,
                        help='Number of cells per sample')
    parser.add_argument('--n_genes', type=int, default=50,
                        help='Number of genes')
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Epsilon-ball radius')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()


    np.random.seed(args.seed)

    generator = SpatiotemporalBifurcationGenerator(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        radius=args.radius
    )


    print(f"Generating {args.samples_per_class * 6} samples (6 bifurcation types)...")
    dataset = generator.generate_dataset(n_samples_per_class=args.samples_per_class,
                                          save_path=None)

    print("Splitting into train/val/test...")
    train_data, val_data, test_data = split_dataset(dataset, 0.7, 0.15)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_dir}...")
    np.savez_compressed(output_dir / 'train.npz', dataset=train_data)
    np.savez_compressed(output_dir / 'val.npz', dataset=val_data)
    np.savez_compressed(output_dir / 'test.npz', dataset=test_data)

    print(f"\n✓ Data generation complete!")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")


if __name__ == '__main__':
    main()
