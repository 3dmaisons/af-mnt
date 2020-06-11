#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES 

# python 3.6
# pytorch 1.1
# source activate pt11-cuda9
source activate py13-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3


gen_mode=afstatic
load_af=models-v9enfr/af-v0001/checkpoints_epoch/70
test_path_out=models-v9enfr/af-v0001/tst2013_klloss/epoch_70/
# gen_mode=afdynamic
# load_af=models-v9enfr/afdynamic-v0002/checkpoints_epoch/8
# test_path_out=models-v9enfr/afdynamic-v0002/tst2013_klloss/epoch_8/
# ---------------------------------------------------------------------------------------------[generate klloss]
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v9/translate.py \
    --test_path_src lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.en \
    --test_path_tgt lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.fr \
    --path_vocab_src lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
    --path_vocab_tgt lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
    --load_tf models-v9enfr/tf-v0001/checkpoints_epoch/24 \
    --load_af $load_af \
    --test_path_out $test_path_out \
    --max_seq_len 20 \
    --batch_size 1 \
    --use_gpu True \
    --beam_width 1 \
    --mode 9 \
    --gen_mode $gen_mode \

# ---------------------------------------------------------------------------------------------[generate txt]
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v9/translate.py \
#     --test_path_src lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.en \
#     --test_path_tgt lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.fr \
#     --path_vocab_src lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
#     --path_vocab_tgt lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
#     --load models-v9enfr/afdynamic-v0002/checkpoints_epoch/8 \
#     --test_path_out models-v9enfr/afdynamic-v0002/tst2013/epoch_8/ \
#     --max_seq_len 200 \
#     --batch_size 64 \
#     --use_gpu False \
#     --beam_width 1 \
#     --use_teacher True \
#     --mode 4 \
    # --test_attscore_path lib/attscores/iwslt.enfr.2012tst.tfv0001.npy \


    # --load models-v9enfr/af-v0001/checkpoints_epoch/70 \
    # --test_path_out models-v9enfr/af-v0001/tst2013/epoch_70/ \
    # --load models-v9enfr/tf-v0001/checkpoints_epoch/24 \
    # --test_path_out models-v9enfr/tf-v0001/tst2013/epoch_24/ \


    # NOTE:
    # for beam search width = 10; batch_size max = 8

    # --test_path_src lib/iwslt15-ytl/tst2012.en \
    # --test_path_tgt lib/iwslt15-ytl/tst2012.vi \
    # --test_path_src lib/iwslt15-ytl/tst2013.en \
    # --test_path_tgt lib/iwslt15-ytl/tst2013.vi \

    # ===========================
    # attention generate
    # --test_path_src lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.en \
    # --test_path_tgt lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.fr \
    # --path_vocab_src lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
    # --path_vocab_tgt lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
    # --load models-v9enfr/tf-v0001/checkpoints_epoch/24 \
    # --test_path_out models-v9enfr/tf-v0001/trainset/epoch_24/ \
    # --max_seq_len 64 \
    # --batch_size 64 \
    # --use_gpu False \
    # --beam_width 1 \
    # --use_teacher True \
    # --mode 6 \

    # ===========================
    # attention generate
    # --test_path_src lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
    # --test_path_tgt lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.fr \
    # --path_vocab_src lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
    # --path_vocab_tgt lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
    # --load models-v9enfr/tf-v0001/checkpoints_epoch/24 \
    # --test_path_out models-v9enfr/tf-v0001/tst2012-attscore/epoch_16/ \
    # --max_seq_len 200 \
    # --batch_size 64 \
    # --use_gpu True \
    # --beam_width 1 \
    # --use_teacher True \
    # --mode 6 \

    # ===========================
    # normal translate 
    # --test_path_src lib/iwslt15-ytl/tst2013.en \
    # --test_path_tgt lib/iwslt15-ytl/tst2013.vi \
    # --path_vocab_src lib/iwslt15-ytl/vocab.en \
    # --path_vocab_tgt lib/iwslt15-ytl/vocab.vi \
    # --load models-v9/envi-v0004/checkpoints_epoch/20 \
    # --test_path_out models-v9/envi-v0004/iwslt13/epoch_20/ \
    # --max_seq_len 200 \
    # --batch_size 32 \
    # --use_gpu True \
    # --beam_width 1 \    
