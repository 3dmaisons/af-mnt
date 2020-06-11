#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
# export CUDA_VISIBLE_DEVICES= #note on nausicaa no.2 is no.0
echo $CUDA_VISIBLE_DEVICES 

# python 3.6
# pytorch 1.1
source activate pt11-cuda9

export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3

for i in `seq 37 1 50`
do 
	echo $i
	$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v9/translate.py \
	    --test_path_src lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
	    --test_path_tgt lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.fr \
	    --path_vocab_src lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
	    --path_vocab_tgt lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
	    --load models-v9enfr/aftf-v0002/checkpoints_epoch/$i \
	    --test_path_out models-v9enfr/aftf-v0002/tst2012/epoch_$i/ \
	    --max_seq_len 200 \
	    --batch_size 64 \
	    --use_gpu True \
	    --beam_width 1 \
	    --use_teacher False \
	    --mode 2  
done


	# ===== enen/enfr =====
	    # --test_path_src lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
	    # --test_path_tgt lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
	    # --path_vocab_src lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
	    # --path_vocab_tgt lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
	    # --load models-v9enfr/dual-enen-v0001/tf/checkpoints_epoch/$i \
	    # --test_path_out models-v9enfr/dual-enen-v0001/tf/tst2012/epoch_$i/ \
	    # --max_seq_len 200 \
	    # --batch_size 64 \
	    # --use_gpu True \
	    # --beam_width 1 \
	    # --use_teacher False \
	    # --mode 2  

	    # afdynamic
	    # --load models-v9enfr/afdynamic-enen-v0001/checkpoints_epoch/$i \
	    # --test_path_out models-v9enfr/afdynamic-enen-v0001/tst2012/epoch_$i/ \


	# ===== dd =====
	    # --test_path_src lib/swbd+clc_corrupt-new/swbd/valid.txt.dsf \
	    # --test_path_tgt lib/swbd+clc_corrupt-new/swbd/valid.txt \
	    # --path_vocab_src lib/vocab/clctotal+swbd.min-count4.en \
	    # --path_vocab_tgt lib/vocab/clctotal+swbd.min-count4.en \
	    # --load models-v9enfr/dual-dd-v0001/tf/checkpoints_epoch/$i \
	    # --test_path_out models-v9enfr/dual-dd-v0001/tf/swbd-valid/epoch_$i/ \
	    # --max_seq_len 200 \
	    # --batch_size 64 \
	    # --use_gpu True \
	    # --beam_width 1 \
	    # --use_teacher False \
	    # --mode 2  

	    # afdynamic
	    # --load models-v9enfr/afdynamic-v0001/checkpoints_epoch/$i \
	    # --test_path_out models-v9enfr/afdynamic-v0001/tst2012/epoch_$i/ \
