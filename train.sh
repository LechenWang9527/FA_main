CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/eurosat_dtd_falp.yaml \
  --is_train 1 \
  2>&1 | tee logs/train_falp_eurosat_dtd_ep5.log

  CUDA_VISIBLE_DEVICES=0 python main.py \
  --config configs/eurosat_dtd_falp.yaml \
  --is_train 0 \
  2>&1 | tee logs/test_falp_eurosat_dtd_ep5.log