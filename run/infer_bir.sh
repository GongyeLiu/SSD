export CUDA_VISIBLE_DEVICES=0


python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation colorization \
  --deg_noise jpeg \
  --exp_name colorization_noisy_jpeg_60

python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation colorization \
  --deg_noise gauss \
  --exp_name colorization_noisy_gaussian


python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation sr_average \
  --def_factor 8. \
  --deg_noise jpeg \
  --exp_name sr_average_8_noisy_jpeg_60

python inference_ir.py \
  --forward_timestep 15 --backward_timestep 85 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation sr_average \
  --def_factor 8. \
  --deg_noise gauss \
  --exp_name sr_average_8_noisy_gaussian