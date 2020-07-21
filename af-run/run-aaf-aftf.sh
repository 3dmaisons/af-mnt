#!/bin/bash
#$ -S /bin/bash

# ------------------------ ENV --------------------------
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

AIR_FORCE_GPU=0
export MANU_CUDA_DEVICE=0 #note on nausicaa no.2 is no.0
# select gpu when not on air
if [[ "$HOSTNAME" != *"air"* ]]  || [ $AIR_FORCE_GPU -eq 1 ]; then
  X_SGE_CUDA_DEVICE=$MANU_CUDA_DEVICE
fi
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo "on $HOSTNAME, using gpu (no nb means cpu) $CUDA_VISIBLE_DEVICES"


# qd212 202004
source /home/mifs/ytl28/anaconda3/etc/profile.d/conda.sh
conda activate py13-cuda9

EXP_DIR=/home/dawna/tts/qd212/models/af
cd $EXP_DIR

export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3


# ------------------------ CONFIG --------------------------
# ------------------------ data --------------------------
task=enfr # enfr ende
testset_fr=tst2013 # tst2013 tst2014
testset_de=tst-COMMON # tst2013 tst2014

### parse config
case $task in
"enfr")
  use_type=word
  train_path_src=af-lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.en
  train_path_tgt=af-lib/iwslt15-enfr/iwslt15_en_fr/train.tags.en-fr.fr
  path_vocab_src=af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.en
  path_vocab_tgt=af-lib/iwslt15-enfr/iwslt15_en_fr/dict.50k.fr

  testset=$testset_fr
  case $testset in
  "tst2014")
    prefix=en-fr-2015
    event=IWSLT16
    ;;
  "tst2013")
    prefix=iwslt15-enfr
    event=IWSLT15
    ;;
  esac
  test_path_src=af-lib/${prefix}/iwslt15_en_fr/${event}.TED.${testset}.en-fr.en
  test_path_tgt=af-lib/${prefix}/iwslt15_en_fr/${event}.TED.${testset}.en-fr.fr
  ;;
"ende")
  use_type=char
  train_path_src=af-lib/mustc-en-de/train/train.BPE.en
  train_path_tgt=af-lib/mustc-en-de/train/train.de
  path_vocab_src=af-lib/mustc-en-de/vocab.en
  path_vocab_tgt=af-lib/mustc-en-de/vocab.de.char.trim

  testset=$testset_de
  test_path_src=af-lib/mustc-en-de/tst-COMMON/tst-COMMON.BPE.en
  test_path_tgt=af-lib/mustc-en-de/tst-COMMON/tst-COMMON.de
  ;;
esac

# ------------------------ model & mode --------------------------
MODE=translate_smooth # train translate translate_smooth
# TRANSLATE_EPOCH=1
smooth_epochs_str=4_6_7 # 3_5_7 3_4_5_6_7 24_28_33

### dir
case $task in
"enfr")
  SAVE_DIR_BASE=results/models-v9enfr
  load_tf=results/models-v9enfr/aaf-v0013-aftf/checkpoints_epoch/28
  train_attscore_path=af-models/tf/trainset/epoch_24/att_score.npy
  ;;
"ende")
  SAVE_DIR_BASE=results/models-v0ende
  load_tf=None
  train_attscore_path=None
  ;;
esac

# learning_rate=0.002
# SAVE_DIR=results/models-v9enfr/aaf-v0012-af-masksrc/
# SAVE_DIR=results/models-v9enfr/aaf-v0013-af-fixkl/
# SAVE_DIR=results/models-v9enfr/aaf-v0013-af-debug/
# SAVE_DIR=results/models-v9enfr/aaf-v0013-af/
# SAVE_DIR=results/models-v9enfr/aaf-v0013-aftf/
# SAVE_DIR=results/models-v9enfr/aaf-v0013-aftf-bs128/
# SAVE_DIR=results/models-v9enfr/aaf-v0013-aftf-pretrain/

# learning_rate=0.008
# SAVE_DIR=results/models-v9enfr/aaf-v0013-aftf-checkRunAway/
# SAVE_DIR=results/models-v9enfr/aaf-v0010-af-checkRunAway/
# CUDA_LAUNCH_BLOCKING=1 # add before pyhton sth.py

learning_rate=0.001
SAVE_DIR=results/models-v9enfr/aaf-v0013-aftf-pretrain-lr${learning_rate}/


# SAVE_DIR=${SAVE_DIR_BASE}/v0001-aftf/


# ------------------------ RUN MODEL --------------------------
case $MODE in
"train")
    echo MODE: train
      # --dev_path_src af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
      # --dev_path_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.fr \
    $PYTHONBIN /home/dawna/tts/qd212/models/af/af-scripts/train.py \
      --train_path_src $train_path_src \
      --train_path_tgt $train_path_tgt \
      --path_vocab_src $path_vocab_src \
      --path_vocab_tgt $path_vocab_tgt \
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
      --batch_size 50 \
      --batch_first True \
      --eval_with_mask False \
      --scheduled_sampling False \
      --embedding_dropout 0.0 \
      --learning_rate $learning_rate \
      --max_grad_norm 1.0 \
      --use_gpu True \
      --checkpoint_every 500 \
      --print_every 200 \
      --num_epochs 30 \
      --train_mode aaf_base \
      --teacher_forcing_ratio 1.0 \
      --attention_forcing True \
      --attention_loss_coeff 10.0 \
      --save $SAVE_DIR \
      --load_tf $load_tf \
      --train_attscore_path $train_attscore_path \
      2>&1 | tee ${EXP_DIR}/${SAVE_DIR}log.txt
      # --num_epochs 50 \
      # --train_attscore_path lib/attscores/iwslt.enfr.tfv0001.npy \
      # --load_tf af-models/tf/checkpoints_epoch/24 \
    ;;
"translate")
trap "exit" INT
for f in ${EXP_DIR}/${SAVE_DIR}/checkpoints_epoch/*; do
    TRANSLATE_EPOCH=$(basename $f)
    test_path_out=${SAVE_DIR}${testset}/epoch_${TRANSLATE_EPOCH}/
    if [ ! -f "${test_path_out}translate.txt" ]; then
    echo MODE: translate, save to $test_path_out
    $PYTHONBIN /home/dawna/tts/qd212/models/af/af-scripts/translate.py \
        --test_path_src $test_path_src \
        --test_path_tgt $test_path_tgt \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --load ${SAVE_DIR}checkpoints_epoch/${TRANSLATE_EPOCH} \
        --test_path_out $test_path_out \
        --max_seq_len 200 \
        --batch_size 50 \
        --use_gpu True \
        --beam_width 1
        # --use_teacher False \
        # --mode 2 \
        # --test_attscore_path af-models/tf/tst2012-attscore/epoch_24/att_score.npy \
    fi
done
    ;;
"translate_smooth")
    test_path_out=${SAVE_DIR}${testset}/epoch_smooth_${smooth_epochs_str}/
    if [ ! -f "${test_path_out}translate.txt" ]; then
    echo MODE: $MODE, save to $test_path_out
    $PYTHONBIN /home/dawna/tts/qd212/models/af/af-scripts/translate.py \
        --test_path_src $test_path_src \
        --test_path_tgt $test_path_tgt \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --load ${SAVE_DIR}checkpoints_epoch/ \
        --test_path_out $test_path_out \
        --max_seq_len 200 \
        --batch_size 50 \
        --use_gpu True \
        --beam_width 1 \
        --smooth_epochs_str $smooth_epochs_str
    fi
    ;;
esac



























