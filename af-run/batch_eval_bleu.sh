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


cd /home/dawna/tts/qd212/models/af/

indir=results/models-v9enfr/tf-v0001/tst2013
refdir=af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.tst2013.en-fr.fr

# indir=results/models-v9enfr/af-v0001/tst2013

# indir=results/models-v9enfr/afdynamic-v0001/tst2013

# indir=results/models-v9enfr/sched-af-v0001/tst2013

# indir=results/models-v9enfr/aaf-v0001-tf/tst2013
# indir=results/models-v9enfr/aaf-v0002-tf/tst2013

# indir=results/models-v9enfr/aaf-v0010-af/tst2013
# indir=results/models-v9enfr/aaf-v0013-af-fixkl/tst2013

# indir=results/models-v9enfr/aaf-v0020-sched-debug/tst2013
# indir=results/models-v9enfr/aaf-v0020-sched-log/tst2013
# indir=results/models-v9enfr/aaf-v0020-sched-fr15/tst2013
# indir=results/models-v9enfr/aaf-v0020-sched-fr0.0/tst2013
indir=results/models-v9enfr/aaf-v0020-sched-fr2.0/tst2013

# indir=results/models-v9enfr/aaf-v0013-aftf/tst2013




for i in 4 10 19
do
	echo epoch $i of $indir
	python /home/dawna/tts/qd212/models/af/bleu_scorer.py $indir/epoch_$i/translate.txt $refdir > $indir/epoch_$i/bleu.log
done


