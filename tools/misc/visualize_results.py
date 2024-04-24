import argparse
import mmcv
from mmcv import Config

from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('--config', default='', help='test config file path')
    parser.add_argument('--result', default='', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', default='', help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)

    if getattr(dataset, 'show', None) is not None:
        dataset.show(results, args.show_dir)
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()
