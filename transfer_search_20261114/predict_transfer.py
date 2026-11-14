"""Predict from cached frozen ResNet features without manual labels or PyTorch."""
from pathlib import Path
import argparse
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.metrics.pairwise import linear_kernel
from face_model import features, flip, unit


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('prediction_transfer.txt'))
    args = parser.parse_args()
    bundle = joblib.load(root / 'model.joblib')
    cfg = bundle['config']
    cache = np.load(root / (cfg['source'] + '.npz'))
    x = loadmat(root.parent / 'test(Task_2)' / 'test_data.mat')['test_data']
    train = bundle['train_features']
    scores = []
    for raw, key in [(x, 'test'), (flip(x), 'test_mirror')]:
        transfer = cache[key]
        assert len(transfer) == len(raw)
        f = transfer if cfg['transfer_weight'] == 0 else unit(np.concatenate([
            features(raw, 'hog_1.0'), cfg['transfer_weight'] * transfer], axis=1))
        cosine = linear_kernel(f, train)
        if cfg['kernel'] == 'rbf':
            kernel = np.exp(-cfg['param'] * np.maximum(2 - 2 * cosine, 0))
        elif cfg['kernel'] == 'poly2':
            kernel = (cosine + 1) ** 2
        else:
            kernel = cosine
        scores.append(bundle['model'].decision_function(kernel))
    score = np.mean(scores, axis=0) if cfg['inference'] == 'mirror_mean' else scores[0]
    pred = np.where(score > cfg['threshold'], 1, -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(''.join(f'{i} {int(v)}\n' for i, v in enumerate(pred, 1)).encode('ascii'))
    print(f'Saved {len(pred)} predictions to {args.output}; face={sum(pred == 1)}, nonface={sum(pred == -1)}')


if __name__ == '__main__':
    main()
