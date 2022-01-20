import yaml
import argparse
from eod.data.metrics.base_evaluator import build_evaluator

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('config', type=str)
    argparser.add_argument('result', type=str)
    args = argparser.parse_args()
    with open(args.config) as fd:
        cfg = yaml.safe_load(fd)
    test_dataset = cfg['dataset']['test']['dataset']
    evaluator = build_evaluator(test_dataset['kwargs']['evaluator'])
    metrics = evaluator.eval(args.result)
    print(metrics)
