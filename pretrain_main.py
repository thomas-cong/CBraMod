import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod
from pretrain_trainer import Trainer
import wandb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=3, help='cuda number (default: 1)')
    parser.add_argument('--parallel', action='store_true', help='parallel')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', action='store_true', default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')

    parser.add_argument('--dataset_dir', type=str, default='dataset_dir',
                        help='dataset_dir')
    parser.add_argument('--model_dir',   type=str,   default='model_dir', help='model_dir')
    parser.add_argument('--use_spectrogram', action='store_true', default=False, help='use_spectrogram')
    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb logging')
    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    
    # Initialize wandb
    run_name = f"CBraMod_B{params.batch_size}_E{params.epochs}_lr{params.lr}_n{params.n_layer}_h{params.nhead}"
    params.run_name = run_name
    if params.use_spectrogram:
        run_name += "_spectrogram"
    if params.wandb:
        wandb.init(
                name=run_name,
                project="cbramod",
                entity='learning_to_adapt', # team entity
                config=vars(params),
                dir=params.model_dir,
                )
    
    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir, use_spectrogram = params.use_spectrogram)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=2,
        shuffle=True,
    )
    model = CBraMod(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
        params.nhead
    )
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    pretrained_dataset.db.close()
    
    # Finish wandb run
    if params.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
