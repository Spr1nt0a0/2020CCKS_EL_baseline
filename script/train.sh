DIR=$(cd $(dirname $0); pwd)
MODEL_PATH="./pretrained_model/ERNIE_1.0_max-len-512"
TASK_DATA_PATH="./data/generated"
export FLAGS_eager_delete_tensor_gb=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/anaconda3/bin/python3.6
python -u /home/baochen/competition/ccs2020_EL/ernie/run_type_pairwise_ranker.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size 32 \
                   --metric "acc_and_f1" \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.txt \
                   --dev_set ${TASK_DATA_PATH}/dev.txt \
                   --use_multi_gpu_test true \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --label_map ${TASK_DATA_PATH}/type_label_map.json \
                   --checkpoints "./checkpoints" \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 1 \
                   --max_seq_len 128 \
                   --learning_rate 2e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1
