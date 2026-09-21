import os
import argparse
import datetime
import torch
import numpy as np
from utils.tools import set_seed, log_to_file
from exp.exp_main import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description='CTIA-ODE for Multivariate Time Series Forecasting')

    parser.add_argument('--root_path', type=str, default='./dataset/weather/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--dataset_name', type=str, default='weather', help='dataset name')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[96, 192, 336, 720],
                        help='prediction sequence lengths')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=12, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    args = parser.parse_args()

    set_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = os.path.join("results", args.dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(results_dir, f"metrics_{current_time}.txt")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {args.dataset_name}\nTime: {current_time}\n")

    all_results = []

    for pred_len in args.pred_lens:
        args.pred_len = pred_len
        print(f">>>>>>> Start training : {args.dataset_name} | Seq: {args.seq_len} | Pred: {args.pred_len} >>>>>>>")

        mse, mae = train_and_evaluate(args, device, log_file_path)
        all_results.append((pred_len, mse, mae))

    log_to_file(log_file_path, "\n" + "=" * 78 + "\nFINAL SUMMARY\n" + "=" * 78)
    for pred_len, mse, mae in all_results:
        log_to_file(log_file_path, f"T={pred_len:<3d} | MSE={mse:.6f} | MAE={mae:.6f}")

    avg_mse = np.mean([res[1] for res in all_results])
    avg_mae = np.mean([res[2] for res in all_results])
    log_to_file(log_file_path, "-" * 78 + f"\nAverage MSE={avg_mse:.6f} | Average MAE={avg_mae:.6f}\n" + "=" * 78)


if __name__ == "__main__":
    main()