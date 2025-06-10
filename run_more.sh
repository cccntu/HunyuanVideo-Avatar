#!/bin/bash

export PYTHONPATH=./
export MODEL_BASE="./weights"
export OUTPUT_BASEPATH="./results_more"
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt

# Generate more examples using the full test.csv (which has 24 examples vs single_test.csv with 1)
echo "Generating more examples using assets/test.csv..."

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
        --input 'assets/test.csv' \
        --ckpt ${checkpoint_path} \
        --sample-n-frames 129 \
        --seed 128 \
        --image-size 512 \
        --cfg-scale 7.5 \
        --infer-steps 10 \
        --use-deepcache 1 \
        --flow-shift-eval-video 5.0 \
        --save-path ${OUTPUT_BASEPATH}

echo "Batch completed! Results saved to ${OUTPUT_BASEPATH}/"