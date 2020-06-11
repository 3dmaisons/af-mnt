#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH
# export PATH=/home/mifs/ytl28/anaconda3/bin/:/home/mifs/ytl28/anaconda/bin:/home/mifs/ytl28/local/bin:/home/mifs/ytl28/anaconda3/condabin:\
# /home/mifs/ytl28/bin:/home/mifs/ytl28/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:$PATH
# export PATH=/home/mifs/ytl28/anaconda3/bin/:/home/mifs/ytl28/anaconda/bin:/home/mifs/ytl28/local/bin:/home/mifs/ytl28/anaconda3/condabin:/home/mifs/ytl28/bin:/home/mifs/ytl28/.local/bin:$PATH

export MANU_CUDA_DEVICE=0 #note on nausicaa no.2 is no.0
# select gpu when not on air
if [[ "$HOSTNAME" != *"air"* ]]; then
  X_SGE_CUDA_DEVICE=$MANU_CUDA_DEVICE
fi
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo "on $HOSTNAME, using gpu (no nb means cpu) $CUDA_VISIBLE_DEVICES"

# python 3.6 
# pytorch 1.1
# source activate pt11-cuda9
# source activate pt12-cuda10

# qd212 202004
# source /home/dawna/tts/qd212/anaconda2/etc/profile.d/conda.sh
# conda activate p37_torch11_cuda9
source /home/mifs/ytl28/anaconda3/etc/profile.d/conda.sh
conda activate py13-cuda9

EXP_DIR=/home/dawna/tts/qd212/models/af
cd $EXP_DIR

# qd212
# export PYTHONPATH=/home/dawna/tts/qd212/anaconda2/envs/p37_torch13_cuda9/lib/python3.7/site-packages/torch:$PYTHONPATH
# python /home/dawna/tts/qd212/models/af/af-scripts/train.py \
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3


MODE=train # train translate
# SAVE_DIR=results/models-v9enfr/aaf-v0003-tf-asup/
SAVE_DIR=results/models-v9enfr/aaf-v0003-tf-checkRunAway/
TRANSLATE_EPOCH=30

case $MODE in
"train")
    echo MODE: train
    CUDA_LAUNCH_BLOCKING=1 $PYTHONBIN /home/dawna/tts/qd212/models/af/af-scripts/train.py \
      --train_path_src af-lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.en \
      --train_path_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.fr \
      --dev_path_src af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
      --dev_path_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.fr \
      --path_vocab_src af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
      --path_vocab_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
      --random_seed 16 \
      --embedding_size_enc 200 \
      --embedding_size_dec 200 \
      --hidden_size_enc 200 \
      --num_bilstm_enc 2 \
      --num_unilstm_enc 0 \
      --hidden_size_dec 200 \
      --num_unilstm_dec 4 \
      --hidden_size_att 10 \
      --hard_att False \
      --att_mode bilinear \
      --residual True \
      --hidden_size_shared 200 \
      --dropout 0.2 \
      --max_seq_len 64 \
      --batch_size 128 \
      --batch_first True \
      --eval_with_mask False \
      --scheduled_sampling False \
      --embedding_dropout 0.0 \
      --learning_rate 0.008 \
      --max_grad_norm 1.0 \
      --use_gpu True \
      --checkpoint_every 500 \
      --print_every 200 \
      --num_epochs 50 \
      --train_mode aaf \
      --teacher_forcing_ratio 1.0 \
      --attention_forcing False \
      --attention_loss_coeff 0.0 \
      --save $SAVE_DIR \
      --train_attscore_path af-models/tf/trainset/epoch_24/att_score.npy \
      2>&1 | tee ${EXP_DIR}/${SAVE_DIR}log.txt
      # --num_epochs 50 \
      # --train_attscore_path lib/attscores/iwslt.enfr.tfv0001.npy \

      # --load_tf af-models/tf/checkpoints_epoch/24 \

      # bahdanau / hybrid
      # --train_path_src lib/iwslt15-ytl/train.en \
      # --train_path_tgt lib/iwslt15-ytl/train.vi \
      # --path_vocab_src lib/iwslt15-ytl/vocab.en \
      # --path_vocab_tgt lib/iwslt15-ytl/vocab.vi \
      # --dev_path_src lib/iwslt15-ytl/tst2012.en \
      # --dev_path_tgt lib/iwslt15-ytl/tst2012.vi \
    ;;
"translate")
    echo MODE: translate with ckpt of epoch ${TRANSLATE_EPOCH}
    $PYTHONBIN /home/dawna/tts/qd212/models/af/af-scripts/translate.py \
        --test_path_src af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.en \
        --test_path_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.fr \
        --path_vocab_src af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en \
        --path_vocab_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr \
        --load ${SAVE_DIR}checkpoints_epoch/${TRANSLATE_EPOCH} \
        --test_path_out ${SAVE_DIR}tst2013/epoch_${TRANSLATE_EPOCH}/ \
        --max_seq_len 200 \
        --batch_size 64 \
        --use_gpu True \
        --beam_width 1 \
        # --use_teacher False \
        # --mode 2 \
        # --test_attscore_path af-models/tf/tst2012-attscore/epoch_24/att_score.npy \
    ;;
esac



























