export CUDA_VISIBLE_DEVICES=0
python inference_ir.py \
  --forward_timestep 5 --backward_timestep 25 --insert_step 750 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.8 \
  --degradation colorization \
  --exp_name celeba_colorization


python inference_ir.py \
  --forward_timestep 5 --backward_timestep 25 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation deblur_gauss \
  --exp_name celeba_deblur


python inference_ir.py \
  --forward_timestep 5 --backward_timestep 25 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation sr_bicubic \
  --def_factor 4. \
  --exp_name celeba_sr_bicubic_4


python inference_ir.py \
  --forward_timestep 5 --backward_timestep 25 --insert_step 550 \
  --save_dir ./data_demo/restored_results/ \
  --cfg_path ./configs/inference/celeba_256.yaml \
  --noise_eta 0.4 \
  --degradation sr_bicubic \
  --def_factor 8. \
  --exp_name celeba_sr_bicubic_8