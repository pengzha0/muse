#!/bin/sh

gpu_id=0,1,2,3,4,5,6,7
continue_from=

if [ -z ${continue_from} ]; then
	log_name='avaNet_'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=8 \
--master_port=1249 \
main.py \
\
--log_name $log_name \
\
--batch_size 8 \
--audio_direc '/data/vgc/users/public/voxceleb2/muse/audio_clean/' \
--visual_direc '/data/vgc/users/public/voxceleb2/muse/lip/' \
--mix_lst_path '/data/vgc/users/public/voxceleb2/uesv/mixture_data_list_2_800mix.csv' \
--mixture_direc '/data/vgc/users/public/voxceleb2/uesv/audio_mixture/2_mix_min_train/' \
--C 2 \
--epochs 20 \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1
