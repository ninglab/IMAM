# README
The PyTorch implementation for $IMAM$

## Environments

- python 3.7.3
- PyTorch (version: 1.4.0)
- numpy (version: 1.16.2)

## Dataset and Data preprocessing

We include the processed Diginetica, Nowplaying and Tmall datasets in the "datasets" folder.

Due to the policy, we cannot share the other datasets. Please refer to the original source as presented in our manuscript for these datasets.

We adapted the scripts from [LESSR](https://github.com/twchen/lessr) for the preprocessing. 

You can find them in the "preprocess" fold

## Example
Please refer to the following example on how to train and evaluate $IMAM\text{-}O$, $IMAM\text{-}P$ and $IMAM\text{-}O\text{-}P$ on Diginetica. 

Please change the value of "data", and the hyper parameters (e.g., max_len for $n$) as presented in our appendix to reproduce the results on the other datasets.

You are recommended to run the code using GPUs.

$IMAM\text{-}O$:

```
python run.py --data=diginetica --n_epoch=60 --dim=128 --max_len=10 --isTrain=0 --model=P2MAMO --num_heads=1
```

$IMAM\text{-}P$:

```
python run.py --data=diginetica --n_epoch=60 --dim=128 --max_len=10 --isTrain=0 --model=P2MAMP --num_heads=1
```

$IMAM\text{-}O\text{-}P$:

```
python run.py --data=diginetica --n_epoch=60 --dim=128 --max_len=15 --isTrain=0 --model=P2MAMOP --num_heads=8
```
