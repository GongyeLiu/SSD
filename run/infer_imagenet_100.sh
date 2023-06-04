export CUDA_VISIBLE_DEVICES=6
python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --aligned_timestep 6 7 8 9 10 11 12 13 \
  --save_dir ./data_demo/restored_results_100/ \
  --cfg_path ./configs/inference/imagenet_256.yaml \
  --noise_eta 0.4 \
  --use_consistency True \
  --degradation deblur_gauss \
  --exp_name imagenet_deblur


python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 750 \
  --aligned_timestep 6 7 8 9 10 11 12 13 \
  --save_dir ./data_demo/restored_results_100/ \
  --cfg_path ./configs/inference/imagenet_256.yaml \
  --noise_eta 0.8 \
  --use_consistency True \
  --degradation colorization \
  --exp_name imagenet_colorization


python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --save_dir ./data_demo/restored_results_100/ \
  --cfg_path ./configs/inference/imagenet_256.yaml \
  --noise_eta 0.4 \
  --use_consistency True \
  --degradation sr_bicubic \
  --factor 4. \
  --exp_name imagenet_sr_bicubic_4


python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --save_dir ./data_demo/restored_results_100/ \
  --cfg_path ./configs/inference/imagenet_256.yaml \
  --noise_eta 0.4 \
  --use_consistency True \
  --degradation sr_bicubic \
  --factor 8. \
  --exp_name imagenet_sr_bicubic_8
