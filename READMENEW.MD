test nyu
```angular2html
python main.py --dir_data ../datas/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
    --test_only --pretrain ../checkpoints/NYUv2.pt --save ../results
```
python main.py --dir_data /mnt/data/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
    --test_only --pretrain /mnt/checkpoints/CFormer/NYUv2.pt --save ../results

python main.py --dir_data /mnt/data/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
    --test_only --pretrain /mnt/checkpoints/CFormer/NYUv2.pt --save ../results

python main.py --dir_data /mnt/data/kitti_depth --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
    --patch_height 240 --patch_width 1216 --gpus 0 --max_depth 90.0 --top_crop 100 --test_crop --save-image \
    --test_only --pretrain /mnt/checkpoints/CFormer/KITTIDC_L1L2.pt --save NAME_TO_SAVE

**num_mask=8**
``` RMSE: 1.30448  MAE: 1.05266  iRMSE: 0.89140  iMAE: 0.24200  REL: 0.34600  D_1: 0.12602 
1
```

```RMSE: 1.24475  MAE: 0.99224  iRMSE: 1.06408  iMAE: 0.23515  REL: 0.32615  D_1: 0.13827
5
```

```RMSE: 1.16729  MAE: 0.91797  iRMSE: 1.06992  iMAE: 0.22579  REL: 0.30301  D_1: 0.15276
10
```

```RMSE: 0.92762  MAE: 0.72534  iRMSE: 0.18961  iMAE: 0.13754  REL: 0.24518  D_1: 0.19335
20
```

```RMSE: 0.71261  MAE: 0.52368  iRMSE: 0.15088  iMAE: 0.10259  REL: 0.18133  D_1: 0.32884
50
```

```RMSE: 0.44249  MAE: 0.26477  iRMSE: 0.08754  iMAE: 0.04866  REL: 0.09218  D_1: 0.65433
100
```

```RMSE: 0.14724  MAE: 0.06344  iRMSE: 0.02406  iMAE: 0.00976  REL: 0.02176  D_1: 0.95180
200
```

```RMSE: 0.09180  MAE: 0.03562  iRMSE: 0.01401  iMAE: 0.00513  REL: 0.01200  D_1: 0.98023
500
```

```RMSE: 0.07101  MAE: 0.02627  iRMSE: 0.01079  iMAE: 0.00374  REL: 0.00882  D_1: 0.98746
1000
```

```RMSE: 0.05554  MAE: 0.01990  iRMSE: 0.00852  iMAE: 0.00283  REL: 0.00669  D_1: 0.99194
2000
```

```RMSE: 0.04517  MAE: 0.01626  iRMSE: 0.00682  iMAE: 0.00233  REL: 0.00548  D_1: 0.99484
5000
```

```RMSE: 0.06568  MAE: 0.02767  iRMSE: 0.00882  iMAE: 0.00391  REL: 0.00941  D_1: 0.99142
10000
```

```RMSE: 0.20324  MAE: 0.13359  iRMSE: 0.02562  iMAE: 0.01810  REL: 0.04611  D_1: 0.88930
20000
```

----------------------mask100---------------------------
```RMSE: 1.05171  MAE: 0.83613  iRMSE: 0.20562  iMAE: 0.15438  REL: 0.28114  D^1: 0.39652  D^2: 0.73807  D^3: 0.90420  D102: 0.03591  D105: 0.08623  D110: 0.16635
10
```

```RMSE: 0.7087  MAE: 0.5211  iRMSE: 0.1455  iMAE: 0.1011  REL: 0.1807  D_1: 0.0647
50
```

```RMSE: 0.4341  MAE: 0.2623  iRMSE: 0.0866  iMAE: 0.0487  REL: 0.0922  D_1: 0.2340
100
```

```RMSE: 0.1423  MAE: 0.0609  iRMSE: 0.0224  iMAE: 0.0091  REL: 0.0206  D_1: 0.6161
200
```

```RMSE: 0.0913  MAE: 0.0354  iRMSE: 0.0139  iMAE: 0.0051  REL: 0.0119  D_1: 0.7577
500
```

```RMSE: 0.0707  MAE: 0.0261  iRMSE: 0.0108  iMAE: 0.0037  REL: 0.0088  D_1: 0.9875
1000
```

```RMSE: 0.0555  MAE: 0.0199  iRMSE: 0.0085  iMAE: 0.0028  REL: 0.0067  D_1: 0.9919
2000
```

```RMSE: 0.0450  MAE: 0.0162  iRMSE: 0.0068  iMAE: 0.0023  REL: 0.0055  D_1: 0.9949
5000
```

```RMSE: 0.0657  MAE: 0.0277  iRMSE: 0.0088  iMAE: 0.0039  REL: 0.0094  D_1: 0.9914
10000
```

```RMSE: 0.2038  MAE: 0.1337  iRMSE: 0.0267  iMAE: 0.0182  REL: 0.0462  D_1: 0.8879
20000
```