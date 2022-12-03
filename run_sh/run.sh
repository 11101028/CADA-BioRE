echo "pid:$$"
gpu_id=0
data_dir=./CMeIE
pretrained_model_name=RoBERTa_zh_Large_PyTorch
cd ../

rdrop_type='sigmoid'
version1=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version_gper:${version1}"
python -u ./code_main/run_gper.py --do_rdrop --do_train --train_batch_size 20 --eval_batch_size 10 --time ${version1} --pretrained_model_name ${pretrained_model_name} --data_dir ${data_dir} --devices ${gpu_id}


version2=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version_re:${version2}"
python -u ./code_main/run_baseline-re.py --do_rdrop --do_train --train_batch_size 22 --eval_batch_size 900 --time ${version2} --pretrained_model_name ${pretrained_model_name} --data_dir ${data_dir} --devices ${gpu_id}


prefix_mode='pm1'
version3=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version_p2so:${version3}"
python -u ./code_main/run_dual-re.py --do_rdrop --rdrop_type ${rdrop_type} --prefix_mode ${prefix_mode} --do_train --train_batch_size 22 --eval_batch_size 10 --time ${version3} --pretrained_model_name ${pretrained_model_name} --data_dir ${data_dir} --devices ${gpu_id}
