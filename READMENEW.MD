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