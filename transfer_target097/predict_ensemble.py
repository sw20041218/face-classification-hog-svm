"""Reconstruct training features and predict the bundled 420 inputs without labels."""
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits


def main():
    threadpool_limits(4)
    folder = Path(__file__).resolve().parent
    root = folder.parent
    cache_dir = root / 'transfer_search_20261114'
    sys.path.insert(0, str(cache_dir))
    from face_model import features, flip, unit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('prediction_transfer097.txt'))
    args = parser.parse_args()
    bundle = joblib.load(folder / 'ensemble_model.joblib')
    x = loadmat(root / 'train(Task_2)' / 'train_data.mat')['train_data']
    test = loadmat(root / 'test(Task_2)' / 'test_data.mat')['test_data']
    base_train = np.concatenate([features(x, 'hog_1.0'), features(flip(x), 'hog_1.0')])
    base_test = features(test, 'hog_1.0')
    base_mirror = features(flip(test), 'hog_1.0')
    fused_score = np.zeros(len(test))
    for member, weight, scale in zip(bundle['members'], bundle['weights'], bundle['score_std']):
        cfg = member['config']
        with np.load(cache_dir / (cfg['source'] + '.npz')) as cache:
            transfer_train = np.concatenate([cache['train'], cache['train_mirror']])
            f = unit(np.concatenate([base_train, cfg['weight'] * transfer_train], axis=1))
            ft = unit(np.concatenate([base_test, cfg['weight'] * cache['test']], axis=1))
            fm = unit(np.concatenate([base_mirror, cfg['weight'] * cache['test_mirror']], axis=1))
        def decision(z):
            kernel = np.exp(-cfg['gamma'] * np.maximum(2 - 2 * (z @ f.T), 0))
            return member['model'].decision_function(kernel)
        score = (decision(ft) + decision(fm)) / 2
        fused_score += weight * score / scale
    pred = np.where(fused_score > bundle['threshold'], 1, -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(''.join(f'{i} {int(p)}\n' for i, p in enumerate(pred, 1)).encode('ascii'))
    print(f'Saved {len(pred)} predictions; face={sum(pred == 1)}, nonface={sum(pred == -1)}')


if __name__ == '__main__':
    main()
