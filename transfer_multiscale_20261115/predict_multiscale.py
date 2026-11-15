"""Predict the existing 420 inputs using the compact multiscale ensemble and caches."""
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
    old = root / 'transfer_search_20261114'
    sys.path.insert(0, str(old))
    from face_model import features, flip, unit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('prediction_multiscale.txt'))
    args = parser.parse_args()
    bundle = joblib.load(folder / 'updated_ensemble_model.joblib')
    x = loadmat(root / 'train(Task_2)' / 'train_data.mat')['train_data']
    test = loadmat(root / 'test(Task_2)' / 'test_data.mat')['test_data']
    bf = np.concatenate([features(x, 'hog_1.0'), features(flip(x), 'hog_1.0')])
    bt, bm = features(test, 'hog_1.0'), features(flip(test), 'hog_1.0')
    keys = ['train', 'train_mirror', 'test', 'test_mirror']
    def source_arrays(source):
        if source == 'resnet50_96_160':
            with np.load(folder / 'resnet50_160.npz') as a, np.load(old / 'resnet50_96.npz') as b:
                return {k: unit(np.concatenate([a[k], b[k]], axis=1)) for k in keys}
        path = folder / (source + '.npz')
        with np.load(path if path.exists() else old / path.name) as a:
            return {k: a[k] for k in keys}
    score = np.zeros(len(test))
    for member, weight, scale in zip(bundle['members'], bundle['weights'], bundle['score_std']):
        cfg = member['config']
        a = source_arrays(cfg['source'])
        transfer = np.concatenate([a['train'], a['train_mirror']])
        w = cfg['weight']
        f = transfer if w == 0 else unit(np.concatenate([bf, w * transfer], axis=1))
        ft = a['test'] if w == 0 else unit(np.concatenate([bt, w * a['test']], axis=1))
        fm = a['test_mirror'] if w == 0 else unit(np.concatenate([bm, w * a['test_mirror']], axis=1))
        def decision(z):
            kernel = np.exp(-cfg['gamma'] * np.maximum(2 - 2 * (z @ f.T), 0))
            return member['model'].decision_function(kernel)
        member_score = decision(ft)
        if cfg['inference'] == 'mirror_mean':
            member_score = (member_score + decision(fm)) / 2
        score += weight * member_score / scale
    pred = np.where(score > bundle['threshold'], 1, -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(''.join(f'{i} {int(p)}\n' for i, p in enumerate(pred, 1)).encode('ascii'))
    print(f'Saved {len(pred)} predictions; face={sum(pred == 1)}, nonface={sum(pred == -1)}')


if __name__ == '__main__':
    main()
