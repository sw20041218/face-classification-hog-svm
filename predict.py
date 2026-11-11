"""Run either saved classifier without training or test annotations."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['baseline', 'shift', 'multiscale'], default='shift')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('prediction.txt'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    folder = root / {'baseline': 'svm_run', 'shift': 'svm_shift_run',
                     'multiscale': 'svm_local_multiscale_run'}[args.experiment]
    sys.path.insert(0, str(folder))
    from face_model import load_data, features, flip
    x, _ = load_data(args.input or root / 'test(Task_2)' / 'test_data.mat')
    if args.experiment == 'baseline':
        model = joblib.load(folder / 'rbf_svm.joblib')
        scores = model.decision_function(x)
        threshold = 0.
    else:
        model = joblib.load(folder / 'model.joblib')
        if args.experiment == 'multiscale':
            from model_features import transform
            cfg = model['config']
            feature_fn = lambda a: transform(a, cfg['mode'], cfg['extra_weight'])
        else:
            feature_fn = lambda a: features(a, model['mode'])
        scores = model['svc'].decision_function(feature_fn(x))
        if model['mirror_tta']:
            scores = (scores + model['svc'].decision_function(feature_fn(flip(x)))) / 2
        threshold = model['threshold']
    pred = np.where(scores > threshold, 1, -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(''.join(f'{i} {int(v)}\n' for i, v in enumerate(pred, 1)).encode('ascii'))
    print(f'Saved {len(pred)} predictions to {args.output}; face={sum(pred == 1)}, nonface={sum(pred == -1)}')


if __name__ == '__main__':
    main()
