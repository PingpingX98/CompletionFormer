
#ckpt=/mnt/checkpoints/CFormer/NYUv2.pt

# for sample in 1 5 50 100 200 300 400 500 
# for sample in 1 50 100 200 500 
# for sample in 1000 5000 20000
# for sample in 20 2000 10000
#for sample in 500

#do
#python main.py --dir_data /mnt/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#    --gpus 0 --max_depth 10.0 --num_sample $sample \
#    --test_only --pretrain $ckpt \
#    --log_dir /mnt/compare/metric/CFormer/experiments/ \
#    --save "test_nyu_sample${sample}_mask8" \
#    --save_result_only \
#    # --save "test_nyu_10msk_sample${sample}" \
#    # --save "test_nyu_8msk_sample${sample}" \
#    # --save 'nyu_1.1' \
#    # --save_full --save_pointcloud_visualization --save_image \
#done
   

python main.py --dir_data ../datas/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 --save_image \
    --test_only --pretrain ../checkpoints/NYUv2.pt --save ../results



#python main.py --dir_data /mnt/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#    --gpus 0 --max_depth 10.0 --num_sample 500 \
#    --test_only --pretrain /mnt/checkpoints/CFormer/NYUv2.pt \
#    --log_dir /mnt/compare/metric/CFormer/experiments/ \
#    --save "test_nyu_sample500_mask8" \
#    --save_result_only