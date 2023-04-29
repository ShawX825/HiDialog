export RoBERTa_LARGE_DIR=/PATH/TO/pre-trained_model/RoBERTa

nohup python run_classifier.py --do_train --do_eval --encoder_type RoBERTa  --data_dir datasets/DialogRE --data_name DialogRE   --vocab_file $RoBERTa_LARGE_DIR/vocab.json --merges_file $RoBERTa_LARGE_DIR/merges.txt  --config_file $RoBERTa_LARGE_DIR/config.json   --init_checkpoint $RoBERTa_LARGE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 12   --learning_rate 7.5e-6   --num_train_epochs 30  --output_dir DialogRE --gradient_accumulation_steps 2 --gpu 0 > logs/DialogRE.log 2>&1 &
