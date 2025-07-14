# we recommend to try this method first for its model accuracy and inference speed
# `--gptq_mse true` may cause overfitting on calibration dataset, if you get a bad quantization result, try to set it `false`
# rotation + gptq 
# activation per-channel quant + weight per-channel quant
model_path=/scratch/janniss/models/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/554bd242505753eef6dfae71f76ddd50c335fc46
tokenizer_path=/scratch/janniss/models/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/554bd242505753eef6dfae71f76ddd50c335fc46
save_path=results
log_name=gemma3-quant

python3 examples/quant_model.py \
--model_path /scratch/janniss/models/hub/models--google--gemma-3-4b-it-qat-int4-unquantized/snapshots/554bd242505753eef6dfae71f76ddd50c335fc46 \
--dtype bfloat16 \
--smooth false \
--rotation true \
--dataset wikitext2 \
--nsamples 128 \
--w_quantizer FixedQuantize \
--w_group_size -1 \
--gptq_mse true \
--gptq_groupsize -1 \
&> gemma3-quant.log