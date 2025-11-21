#!/usr/bin/env python
"""
Grid search for LSTM hyperparameters on mup_scen1,
mirroring run_all_single_LSTM.sh + run_LSTM.sh.

- dataset      : mup_scen1
- LSTM_SIZE    : 50, 100, 150
- N_LAYERS     : 1, 2, 3
- BATCH_SIZE   : 8, 16, 32
- ACTIVATION   : relu, linear
- OPTIMIZER    : rmsprop, adam
- dropout      : 0.15 (fixed, as in run_LSTM.sh)
- learning_rate: 0.001 (fixed, as in run_LSTM.sh)

For each combo it runs:
    python train_LSTM.py    ...
    python evaluate_LSTM.py ...

Results go into ../results/ as final_results_lstm_mup_scen1_*.csv
"""

import itertools
import subprocess

DATASET = "mup_scen1"

LSTM_SIZES  = [50]#, 100, 150]
N_LAYERS    = [1, 2]#, 3]
BATCH_SIZES = [8]#, 16, 32]
ACTIVATIONS = ["relu"]#, "linear"]
OPTIMIZERS  = ["rmsprop", "adam"]

DROPOUT       = 0.15
LEARNING_RATE = 0.001


def run_cmd(cmd_list):
    """Helper to print and run a command."""
    print("\n>>>", " ".join(str(c) for c in cmd_list))
    subprocess.run(cmd_list, check=True)


def main():
    combos = list(itertools.product(
        LSTM_SIZES,
        N_LAYERS,
        BATCH_SIZES,
        ACTIVATIONS,
        OPTIMIZERS,
    ))
    print(f"Total LSTM configs to run: {len(combos)}")

    for lstm_size, n_layers, batch_size, activation, optimizer in combos:
        print("\n============================================================")
        print(f"Config: lstm_size={lstm_size}, n_layers={n_layers}, "
              f"batch_size={batch_size}, activation={activation}, "
              f"optimizer={optimizer}")
        print("============================================================")

        # 1) Train
        train_cmd = [
            "python",
            "train_LSTM.py",
            DATASET,
            str(lstm_size),
            str(DROPOUT),
            str(n_layers),
            str(batch_size),
            str(LEARNING_RATE),
            activation,
            optimizer,
        ]
        run_cmd(train_cmd)

        # 2) Evaluate (MAE/RMSE per prefix length)
        eval_cmd = [
            "python",
            "evaluate_LSTM.py",
            DATASET,
            str(lstm_size),
            str(DROPOUT),
            str(n_layers),
            str(batch_size),
            str(LEARNING_RATE),
            activation,
            optimizer,
        ]
        run_cmd(eval_cmd)


if __name__ == "__main__":
    main()
