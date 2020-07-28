#!/bin/bash
#$ -S /bin/bash

# ------------------------ ENV --------------------------
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
MODE=translate # train translate translate_smooth
smooth_epochs_str=9_11_13 # 9_13_17 5_11_20 13_25_27

### training
learning_rate=0.001 # 0.002

### dir & task-specific setting
case $task in
"enfr")
  SAVE_DIR_BASE=results/models-v9enfr
  load_tf=results/models-v9enfr/aaf-v0002-tf-bs50-v2/checkpoints_epoch/17
  train_attscore_path=af-models/tf/trainset/epoch_24/att_score.npy
  if [ "$MODE" == "train" ]; then
    max_seq_len=64
  else
    max_seq_len=200
  fi
  ;;
"ende")
  SAVE_DIR_BASE=results/models-v0ende
  load_tf=None
  if [ "$MODE" == "train" ]; then
    max_seq_len=300
  else
    max_seq_len=900
  fi
  echo $max_seq_len
  ;;
esac

attention_loss_coeff=1.0 # 10.0 50.0 5.0 2.0 1.0 0.5
teacher_forcing_ratio=0.8 # 1.0 0.8 0.5 0.2

scheduled_sampling=False
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf/
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-notf/
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-asup-detach/
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-asup-sched/
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-tf/
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-gamma${attention_loss_coeff}/

# train_mode=oaf
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-tf${teacher_forcing_ratio}/

train_mode=oaf_alwaysKLsmooth
SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-tf${teacher_forcing_ratio}-alwaysKLsmooth${attention_loss_coeff}/

# train_mode=oaf_alwaysKL
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-tf${teacher_forcing_ratio}-alwaysKL/

# train_mode=oaf_noKL
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-tf${teacher_forcing_ratio}-noKL/

# scheduled_sampling=True
# SAVE_DIR=${SAVE_DIR_BASE}/aaf-v0050-oaf-sched/




# ------------------------ RUN MODEL --------------------------
case $MODE in
"train")
    echo MODE: train
    # --dev_path_src af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.en \
    # --dev_path_tgt af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2012.en-fr.fr \
    # CUDA_LAUNCH_BLOCKING=1
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
      --max_seq_len $max_seq_len \
      --batch_size 50 \
      --batch_first True \
      --eval_with_mask False \
      --scheduled_sampling $scheduled_sampling \
      --embedding_dropout 0.0 \
      --learning_rate $learning_rate \
      --max_grad_norm 1.0 \
      --use_gpu True \
      --checkpoint_every 500 \
      --print_every 200 \
      --num_epochs 30 \
      --train_mode $train_mode \
      --teacher_forcing_ratio $teacher_forcing_ratio \
      --attention_forcing True \
      --attention_loss_coeff $attention_loss_coeff \
      --use_type $use_type \
      --save $SAVE_DIR \
      --load_tf $load_tf \
      2>&1 | tee ${EXP_DIR}/${SAVE_DIR}log.txt
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
        --max_seq_len $max_seq_len \
        --batch_size 50 \
        --use_gpu True \
        --beam_width 1 \
        --use_type $use_type
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
        --load ${SAVE_DIR}checkpoints_epoch/${TRANSLATE_EPOCH} \
        --test_path_out $test_path_out \
        --max_seq_len $max_seq_len \
        --batch_size 50 \
        --use_gpu True \
        --beam_width 1 \
        --smooth_epochs_str $smooth_epochs_str \
        --use_type $use_type
    fi
    ;;
esac



























