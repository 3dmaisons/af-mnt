#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES 

# python 3.6
# pytorch 1.1
source activate pt11-cuda9

export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v9/translate.py \
    --test_path_src lib/iwslt15-ytl/tst2013.en \
    --test_path_tgt lib/iwslt15-ytl/tst2013.vi \
    --path_vocab_src lib/iwslt15-ytl/vocab.en \
    --path_vocab_tgt lib/iwslt15-ytl/vocab.vi \
    --load models-v9/envi-att-v0004/checkpoints_epoch/20 \
    --test_path_out models-v9/envi-att-v0004/debug_noteacher_refattention/epoch_20_plot/ \
    --max_seq_len 200 \
    --batch_size 3 \
    --use_gpu False \
    --use_teacher False \
    --beam_width 1 \
    --mode 4 \
    --test_attscore_path lib/attscores/iwslt.test13.envi.npy \
    # --test_attscore_path lib/attscores/iwslt.test13.envi.noteacher.npy \
 
    # envi-att-v0005-8: use noteacher att
    # envi-att-v0001-4: use withteacher att




