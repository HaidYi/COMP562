import os
import argparse
import torch
import yaml
import logging
import traceback
import numpy as np

from wgan_runner import Wgan_runner

def parser_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='path to the config file',
                        type=str,
                        default='cifar.yml')
    parser.add_argument('--test',
                        help='whether to test the model.',
                        type=bool,
                        default=False)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--image_path',
                        help='the path to the generated images.',
                        type=str,
                        default='./samples')
    parser.add_argument('--resume_training',
                        help='whether to resume training.',
                        type=bool,
                        default=False)
    args = parser.parse_args()

    return args

def parser_config(config_file):
    with open(os.path.join('configs', config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def dict2namespace(dict_):
        namespace = argparse.Namespace()
        for key, value in dict_.items():
            if isinstance(value, dict):
                value = dict2namespace(value)
            setattr(namespace, key, value)
        return namespace

    config = dict2namespace(config)
    return config

def main():
    args = parser_command()
    config = parser_config(args.config)

    # set random seed
    np.random.seed(config.training.np_seed)
    torch.manual_seed(config.training.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.torch_seed)
    torch.backends.cudnn.benchmark = True
    
    # print training configs
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Config =")
    print('\n'.join(['>'*80, str(config), '<'*80]))

    try:
        wgan_runner = Wgan_runner(config, args)
        if not args.test:
            wgan_runner.train()
        wgan_runner.sample(args.sample_size)
    
    except:
        logging.error(traceback.format_exc())

    return 0

if __name__ == '__main__':
    main()