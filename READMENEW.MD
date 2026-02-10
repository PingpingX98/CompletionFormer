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