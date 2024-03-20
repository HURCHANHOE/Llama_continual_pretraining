# pip3 install -r requirements.txt

echo ===== PRETRAINING =====
# Node that MODEL_PATH can be local folder path
# MODEL_PATH=/home/reacubeth/models/llama-7b
# MODEL_PATH=meta-llama/Llama-2-7b
MODEL_PATH=/mnt/b/pretrain/Llama-2-7b-hf
TITLE=llama-7b-pretrain
# DATA=ori_data
DATA=data


OUTPUT_DIR=result
mkdir $OUTPUT_DIR

echo ===== current OUTPUT_DIR is $OUTPUT_DIR =====
echo ===== MODEL_PATH is $MODEL_PATH =====

# node 수에 따라 변경
torchrun --nproc_per_node=4 --master_port=9919 pretrain_hch.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA \
    --bf16 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1  \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False \
    --fp16 True 

#     # gpu가 v100인경우 bf16 tf32 True 옵션 작동안함


# torchrun --nproc_per_node=4 --master_port=9919 pretrain.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --logging_steps 50 \
#     --save_steps 100 \
#     --save_total_limit 1 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
