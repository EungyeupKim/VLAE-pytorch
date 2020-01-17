import argparse
import sys
import os
import torch
from torch import optim
from disvae.training import Trainer
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.vae import VLAE
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_directory, set_seed)
from utils.visualize import GifTraversalsTraining

RES_DIR = "results"

def parse_arguments(args_to_parse):

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description)

    # General options
    parser.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    parser.add_argument('--no-progress-bar', action='store_true',
                         default=False,
                         help='Disables progress bar.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='the number of gpu to use')
    parser.add_argument('-s', '--seed', type=int, default=1234,
                         help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('--viz_single', action='store_true', default=False, help='vizualize with single dim or pair?')

    # Learning options
    parser.add_argument('--checkpoint-every', type=int, default=10,
                          help='Save a checkpoint of the trained model every n epoch.')
    parser.add_argument('-d', '--dataset', help="Path to training data.", default='mnist', choices=DATASETS)
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Maximum number of epochs to run for.')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')

    # Model Options
    parser.add_argument('-z', '--latent_dim', nargs='+', type=int, required=True, help='Dimension of the latent variable.')
    parser.add_argument('-a', '--reg-anneal', type=float, default=10000,
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    parser.add_argument('--reg_coeff0', type=float, default=1.0)
    parser.add_argument('--reg_coeff1', type=float, default=1.0)
    parser.add_argument('--reg_coeff2', type=float, default=1.0)

    # Learning options
    # evaluation = parser.add_argument_group('Evaluation specific options')
    parser.add_argument('--is-eval-only', action='store_true',
                            default=False,
                            help='Whether to only evaluate using precomputed model `name`.')
    # evaluation.add_argument('--is-metrics', action='store_true',
    #                         default=default_config['is_metrics'],
    #                         help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    # evaluation.add_argument('--no-test', action='store_true',
    #                         default=default_config['no_test'],
    #                         help="Whether not to compute the test losses.`")
    parser.add_argument('--eval-batchsize', type=int,
                            default=1000,
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)

    return args

def main(args):

    set_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(RES_DIR, args.name)
    print("save and load experiments at : {}".format(exp_dir))

    if not args.is_eval_only: #train

        create_directory(exp_dir)

        # PREPARES TRAINING DATA
        train_loader = get_dataloaders(args.dataset, batch_size=args.batch_size)

        ##############
        # PREPARES MODEL
        args.img_size = get_img_size(args.dataset)  # stores for metadata
        cs = [1, 64, 128, 1024]
        model = VLAE(args, args.latent_dim, cs)

        #TRAINS
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)  # make sure trainer and viz on same device

        gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)

        reg_coeff = [args.reg_coeff0, args.reg_coeff1, args.reg_coeff2]

        trainer = Trainer(model, optimizer, reg_coeff,
                          device=device,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          gif_visualizer=gif_visualizer)

        trainer(args, train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every)

        #SAVE MODEL AND EXPERIMENT INFORMATION

        save_model(trainer.model, exp_dir, metadata=vars(args))
        print("Model has been saved")
    #
    # if args.is_metrics or not args.no_test:
    #     model = load_model(exp_dir, is_gpu=not args.no_cuda)
    #     metadata = load_metadata(exp_dir)
    #     # TO-DO: currently uses train datatset
    #     test_loader = get_dataloaders(metadata["dataset"],
    #                                   batch_size=args.eval_batchsize,
    #                                   shuffle=False)
    #                                   # logger=logger)
    #
    #     evaluator = Evaluator(model,
    #                           device=device,
    #                           # logger=logger,
    #                           save_dir=exp_dir,
    #                           is_progress_bar=not args.no_progress_bar)
    #
    #     evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
