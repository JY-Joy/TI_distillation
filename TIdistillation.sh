export MODEL_NAME="/share2/huangrenyuan/model_zoo/stable-diffusion-2-1-base"
export DATA_DIR="./data/ip/starry_night"

accelerate launch --num_processes=1 --gpu_ids 1 src/textual_inversion_distillation.py \
  --logging_dir "/share2/haungrenyuan/tb_logs" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --resolution=512 \
  --learning_rate=2.0e-3 \
  --unet_learning_rate=1.0e-06 \
  --seed=3467 \
  --mixed_precision fp16 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --learnable_property="object" \
  --placeholder_token="<painting>" \
  --initializer_token="painting" \
  --validation_steps 500 \
  --validation_prompt "a photo of a <painting>" \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 2000 \
  --resume_from_checkpoint latest \
  --output_dir="distillation/starry_night_testGnorm" \
  --unet_path="/share2/huangrenyuan/model_zoo/bk-sdm-v2-med"
  --num_vectors 2 \
  --push_to_hub