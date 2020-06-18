#!/bin/bash

# Batch evaluate bleu score

# command="$0 $@"
# cmddir=CMDs
# echo "---------------------------------------------" >> $cmddir/batch_eval_bleu.cmds
# echo $command >> $cmddir/batch_eval_bleu.cmds

# indir=$1 # ./models-v9new/envi-v0011/iwslt12
# refdir=$2 # ./lib/iwslt15-ytl/tst2012.vi

# for i in `seq 37 1 50`
# do
#         echo $i
# 	python ./local/py-tools/bleu_scorer.py $1/epoch_$i/translate.txt $2 > $1/epoch_$i/bleu.log
# done

EXP_DIR=/home/dawna/tts/qd212/models/af
cd $EXP_DIR


# 1.0 setup tools
SCRIPTS=/home/dawna/tts/qd212/models/af/af-lib/mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
LANGUAGE=fr
BLEU_DETOK=$SCRIPTS/generic/multi-bleu-detok.perl


# 1.1 select testset

# prefix=iwslt15-enfr
# event=IWSLT15
# testset=tst2013

# # prefix=en-fr-2015
# # event=IWSLT16
# # testset=tst2014

# # refdir=af-lib/${prefix}/iwslt15_en_fr/${event}.TED.${testset}.en-fr.fr
# refdir=af-lib/${prefix}/iwslt15_en_fr/${event}.TED.${testset}.en-fr.DETOK.fr


testset=tst2013
refdir=af-lib/en-fr-2015/iwslt15_en_fr/IWSLT15.TED.${testset}.en-fr.DETOK.fr

# refdir=af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.${testset}.en-fr.fr
# refdir=af-lib/en-fr-2015/iwslt15_en_fr/IWSLT15.TED.${testset}.en-fr.fr


# testset=tst2014
# refdir=af-lib/en-fr-2015/iwslt15_en_fr/IWSLT16.TED.${testset}.en-fr.DETOK.fr






# 1.2 select model

# indir=results/models-v9enfr/tf-v0001/${testset}
# indir=results/models-v9enfr/af-v0001/${testset}
# indir=results/models-v9enfr/afdynamic-v0001/${testset}
# indir=results/models-v9enfr/sched-af-v0001/${testset}

# indir=results/models-v9enfr/aaf-v0001-tf/${testset}
indir=results/models-v9enfr/aaf-v0002-tf/${testset}
# indir=results/models-v9enfr/aaf-v0002-tf-bs50-v2/${testset}

# indir=results/models-v9enfr/aaf-v0010-af/${testset}
# indir=results/models-v9enfr/aaf-v0013-af-fixkl/${testset}
# indir=results/models-v9enfr/aaf-v0013-aftf/${testset}
# indir=results/models-v9enfr/aaf-v0013-aftf-bs128/${testset}

# indir=results/models-v9enfr/aaf-v0020-sched-debug/${testset}
# indir=results/models-v9enfr/aaf-v0020-sched-log/${testset}
# indir=results/models-v9enfr/aaf-v0020-sched-fr15/${testset}
# indir=results/models-v9enfr/aaf-v0020-sched-fr0.0/${testset}
# indir=results/models-v9enfr/aaf-v0020-sched-fr2.0/${testset}
# indir=results/models-v9enfr/aaf-v0020-sched-fr3.25/${testset}
# indir=results/models-v9enfr/aaf-v0020-sched-fr4.0/${testset}


# 2.0 detok and bleu
FILE_TXT=translate-DETOK.txt
FILE_BLEU=bleu-DETOK.log

trap "exit" INT
for d in ${EXP_DIR}/${indir}/*; do
	if [ ! -f ${d}/${FILE_TXT} ]; then
	echo detok, saving to ${d}/${FILE_TXT}
	perl ${DETOKENIZER} -l ${LANGUAGE} < ${d}/translate.txt > ${d}/translate-DETOK.txt
	fi
	if [ ! -f ${d}/${FILE_BLEU} ]; then
	echo BLEU score, saving to ${d}/${FILE_BLEU}
	perl ${BLEU_DETOK} ${refdir} < ${d}/${FILE_TXT} > ${d}/${FILE_BLEU}
	fi
done


# previous version of BLEU scorer, taking tokenized txt as input

# FILE_TXT=translate.txt
# FILE_BLEU=bleu.log
# for d in ${EXP_DIR}/${indir}/*; do
# 	if [ ! -f ${d}/${FILE_BLEU} ]; then
# 	echo saving to $d/${FILE_BLEU}
# 	python /home/dawna/tts/qd212/models/af/bleu_scorer.py $d/${FILE_TXT} $refdir > $d/${FILE_BLEU}
# 	fi
# done


# bkup

# # for d in ${EXP_DIR}/${indir}/*; do
# # 	if [ ! -f ${d}/translate-DETOK.txt ]; then
# # 	echo dir $d
# # 	perl ${DETOKENIZER} -l ${LANGUAGE} < ${d}/translate.txt > ${d}/translate-DETOK.txt
# # 	fi
# # done

# FILE_TXT=translate-DETOK.txt
# FILE_BLEU=bleu-DETOK.log
# for d in ${EXP_DIR}/${indir}/*; do
# 	# if [ ! -f ${d}/${FILE_BLEU} ]; then
# 	echo saving to ${d}/${FILE_BLEU}
# 	perl ${BLEU_DETOK} ${refdir} < ${d}/${FILE_TXT} > ${d}/${FILE_BLEU}
# 	# fi
# done


# for i in 4 10 20 30
# do
# 	echo epoch $i of $indir
# 	python /home/dawna/tts/qd212/models/af/bleu_scorer.py $indir/epoch_$i/translate.txt $refdir > $indir/epoch_$i/bleu.log
# done

