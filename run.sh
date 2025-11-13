#!/bin/bash
# ===== Wan2.1 视频生成脚本 =====

# python run_diffuser_vio.py \
#   --prompt_file ./config/phygen/prompt_force.txt \
#   --vio_prompt_file ./config/phygen/negative_force.txt \
#   --output_dir ./phygen_w10 \
#   --num_frames 25 \
#   --vio_scale 30.0 \
#   --start_num 0

python run_diffuser_vio.py \
  --prompt_file ./config/phygen/prompt_light.txt \
  --vio_prompt_file ./config/phygen/negative_light_GPT.txt \
  --output_dir ./phygen_w10 \
  --num_frames 25 \
  --vio_scale 30.0 \
  --start_num 40

python run_diffuser_vio.py \
  --prompt_file ./config/phygen/prompt_heat.txt \
  --vio_prompt_file ./config/phygen/heat-GPT-V2.txt \
  --output_dir ./phygen_w10 \
  --num_frames 25 \
  --vio_scale 30.0 \
  --start_num 90

python run_diffuser_vio.py \
  --prompt_file ./config/phygen/prompt_material.txt \
  --vio_prompt_file ./config/phygen/negative_material.txt \
  --output_dir ./phygen_w10 \
  --num_frames 25 \
  --vio_scale 30.0 \
  --start_num 120

# python run_diffuser_vio.py \
#   --prompt_file ./config/videophy/fluid_fluid.txt \
#   --vio_prompt_file ./config/videophy/videophy-ff-GPT.txt \
#   --output_dir ./videophy_output_30_ff \
#   --num_frames 25 \
#   --vio_scale 30.0 \
#   --start_num 0

# python run_diffuser_vio.py \
#   --prompt_file ./config/videophy/solid_fluid.txt \
#   --vio_prompt_file ./config/videophy/videophy-sf-GPT.txt \
#   --output_dir ./videophy_output_30_sf \
#   --num_frames 25 \
#   --vio_scale 30.0 \
#   --start_num 55

# python run_diffuser_vio.py \
#   --prompt_file ./config/videophy/solid_solid.txt \
#   --vio_prompt_file ./config/videophy/videophy-ss-GPT.txt \
#   --output_dir ./videophy_output_30_ss \
#   --num_frames 25 \
#   --vio_scale 30.0 \
#   --start_num 201

python run_diffuser.py \
  --prompt_file ./config/phygen/prompt_force.txt \
  --output_dir ./phygen_ori_w10 \
  --num_frames 25
  --start_num 0

python run_diffuser.py \
  --prompt_file ./config/phygen/prompt_light.txt \
  --output_dir ./phygen_ori_w10 \
  --num_frames 25
  --start_num 40

python run_diffuser.py \
  --prompt_file ./config/phygen/prompt_heat.txt \
  --output_dir ./phygen_ori_w10 \
  --num_frames 25
  --start_num 90

python run_diffuser.py \
  --prompt_file ./config/phygen/prompt_material.txt \
  --output_dir ./phygen_ori_w10 \
  --num_frames 25
  --start_num 120

# python run_diffuser.py \
#   --prompt_file ./config/videophy/fluid_fluid.txt \
#   --output_dir ./videophy_ff_ori \
#   --num_frames 25
#   --start_num 0

# python run_diffuser.py \
#   --prompt_file ./config/videophy/solid_fluid.txt \
#   --output_dir ./videophy_sf_ori \
#   --num_frames 25
#   --start_num 55

# python run_diffuser.py \
#   --prompt_file ./config/videophy/solid_solid.txt \
#   --output_dir ./videophy_ss_ori \
#   --num_frames 25
#   --start_num 201