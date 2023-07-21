import argparse

def config():
    parser = argparse.ArgumentParser('KD')
    parser.add_argument('--trial', type=str,
                        help='Exp description.')
    parser.add_argument('--cancer_type', type=str, default='ACC',
                        help='The type of the cancer.(PAAD, etc.)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for reasoning and distillation.')
    parser.add_argument('--lr_final', type=float, default=1e-4,
                        help='Learning rate for final training.')
    parser.add_argument('--reg_scale', type=float, default=0.1,
                        help='Reg_scale for training.')
    parser.add_argument('--dim', type=int, default=5796,
                        help='Input Dimension.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csv_folder', type=str, default=r'./data',
                        help='Path to the csv files.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to the folder the model parameters is loaded from.')
    parser.add_argument('--output_folder', type=str, default='./pth',
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--folder', type=str, default=r'./data/cancer_gene',
                        help='Path to the data. (saved as pickle)')
    parser.add_argument('--mode', type=str, default='all',
                        help='all/few-shot')
    parser.add_argument('--batch_size', type=int, default='10')


    args = parser.parse_args()
    return args
