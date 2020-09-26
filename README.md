# Attention Forcing for Machine Translation

Attention Forcing for Machine Translation, using the standard encoder-decoder NMT model (following Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation Y. Wu et el)

## Prerequisites

- python 3.6
- torch 1.2
- tensorboard 1.14+
- psutil
- dill
- CUDA 9

## Data

- Source / target files: one sentence per line
- Source / target vocab files: one vocab per line, the top 5 fixed to be
`<pad> <unk> <s> </s> <spc>` as defined in `utils/config.py`

The English-French data set is provided in this repository.

To do English-Vietnamese translation, download the data from [the IWSLT 2015 website](https://wit3.fbk.eu/mt.php?release=2015-01).  
In the following scripts, set
- `task` = envi
- `LANGUAGE` = vi


## Train

To train the teacher forcing model - check `af-run/run-tf.sh`

The teacher forcing model can be used to generate the reference attention.
This can be done by setting
- `MODE` = gen_att
- `TRANSLATE_EPOCH` = an epoch, i.e. checkpoint, where the performance is good  
(format example TRANSLATE_EPOCH=17)

To train the vanilla attention forcing model - check `af-run/run-vaf.sh`  
To train the automatic attention forcing model - check `af-run/run-aaf.sh`  
To speed up training, it is strongly recommended to start from a pretrained model.
This can be done by setting
- `load_tf` = an epoch, i.e. checkpoint, where the performance is good  
(format example load_tf=results/models-v9enfr/aaf-v0002-tf-bs50-v2/checkpoints_epoch/17)


## Test

To test a model - check the script used for its training.
Set
- `MODE` = translate
- `testset_fr` or `testset_vi` to the test set you want to use  
Run the script, and it will generate translations with all the checkpoints.

To BLEU score the translations - check `af-run/batch_eval_bleu.sh`  
Set
- `indir` to the path of the translations  
(format example indir=results/models-v0en${LANGUAGE}/v0002-aaf-fr3.5-pretrain-lr0.001-seed4/${testset})
