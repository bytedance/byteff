# Training

## Training on single GPU
```bash
python3 train_local.py --conf $conf
```
`$conf` specifies the path to configuration file.

There are three optional arguments:
- `--timestamp`: with this argument, the training log directory will be named with a timestamp surfix.
- `--only_bond_angle`: with this argument, only the parameters of bond and angle output layers will be trained (used in finetuning).
- `--only_bonded`: with this argument, only the parameters of output layers for bonded parameters will be trained (used in finetuning).

## Training on multiple GPUs
```bash
python3 train_ddp.py --conf $conf
```
`$conf` specifies the path to configuration file.

There are three optional arguments:
- `--timestamp`: with this argument, the training log directory will be named with a timestamp surfix.
- `--restart`: with this argument, training will restart from the newest check point. This argument is not compatible with `--timestamp`.
- `--partial_train`: enable training of some parameters in the entire model (only for debug).