GRU_iters=5
test_augment=0
optim_layer_input_clamp=100.0
depth_activation_format='linear'
depth_downsample_method='min'
pred_confidence_input=1


ckpt=/home/descfly/Projects/CompletionFormer-main/KITTIDC_L1L2.pt
# for lidar_lines in 4 8 16 32 64 
for lidar_lines in 64 
do
  python main_memory.py --dir_data /home/descfly/data/kitti_depth \
      --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
      --patch_height 352 --patch_width 1216 --gpus 0 --max_depth 90.0 \
      --top_crop 0 --test_crop  --test_only \
      --pretrain $ckpt  --lidar_lines $lidar_lines --batch_size 1 \
      --log_dir /data/compare/metric/CFormer/experiments/ \
      --save "val_kitti_lines${lidar_lines}" \
      --save_image
      # --save_result_only 
done
