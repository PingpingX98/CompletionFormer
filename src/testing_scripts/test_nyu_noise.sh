
ckpt=../checkpoints/NYUv2.pt

# for sample in 1 5 50 100 200 300 400 500 
# for sample in 1 50 100 200 500 
# for sample in 1000 5000 20000
# for sample in 20 2000 10000
for noise_level in 0.05 0.04 0.03 0.02 0.01
num_masks=1
sample=500
# do
# # python main.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
# #     --gpus 0 --max_depth 10.0 --num_sample $sample \
# #     --test_only --pretrain $ckpt \
# #     --log_dir /data/compare/metric/CFormer/experiments/ \
# #     --save "test_nyu_8msk_sample${sample}" \
# #     --save_result_only \
# #     # --save "test_nyu_10msk_sample${sample}" \
# #     # --save "test_nyu_8msk_sample${sample}" \
# #     # --save 'nyu_1.1' \
# #     # --save_full --save_pointcloud_visualization --save_image \
# # done
   
do
    python main.py --dir_data /home/descfly/data/nyudepthv2 \
        --data_name NYU --split_json ../data_json/nyu.json \
        --gpus 0 --max_depth 10.0 --num_sample $sample \
        --test_only --pretrain $ckpt  --batch_size 1 \
        --log_dir ../experiments/noise_level(${noise_level}) \
        --noise_level $noise_level
    # --save_result_only
    # --save 'nyu_1.10' \
done