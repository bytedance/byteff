# Data Preparation

## Hessian data
```bash
OMP_NUM_THREADS=1 \
python3 prepare_shards_partial_hessian.py \
    --num_workers $num_workers --shards $shards \
    --meta_file $meta_file --save_path $save_path
```
`$num_workers` specifies the number of processes to run in parallel in a job.\
`$shards` specifies the number of shards the prepared data are saved.\
`$meta_file` specifies the path to the meta_file for the raw data to be converted.\
`$save_path` specifies the path to save the converted data.

## Torsion data
```bash
python3 prepare_shards_torsion.py \
    --num_workers $num_workers --shards $shards \
    --meta_file $meta_file --save_path $save_path
```
Arguments are similar to those for hessian data.

## Energy force data
```bash
python3 prepare_shards_energyforce.py \
    --num_workers $num_workers --shards $shards \
    --meta_file $meta_file --save_path $save_path
```
Arguments are similar to those for hessian data.
