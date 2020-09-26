#!/bin/bash

# Batch evaluate bleu score

EXP_DIR=/home/dawna/tts/qd212/models/af
cd $EXP_DIR

# 1.0 setup tools
SCRIPTS=/home/dawna/tts/qd212/models/af/af-lib/mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
BLEU_DETOK=$SCRIPTS/generic/multi-bleu-detok.perl

# 1.1 select tgt language and testset
LANGUAGE=fr # fr de vi
testset_fr=tst2013 # tst2013 tst2014
testset_vi=tst2012 # tst2012 tst2013

case $LANGUAGE in
"fr")
	testset=$testset_fr
	case $testset in
	"tst2014")
		refdir=af-lib/en-fr-2015/iwslt15_en_fr/IWSLT16.TED.${testset}.en-fr.DETOK.fr
		;;
	"tst2013")
		refdir=af-lib/en-fr-2015/iwslt15_en_fr/IWSLT15.TED.${testset}.en-fr.DETOK.fr
		# refdir=af-lib/iwslt15-enfr/iwslt15_en_fr/IWSLT15.TED.${testset}.en-fr.fr
		# refdir=af-lib/en-fr-2015/iwslt15_en_fr/IWSLT15.TED.${testset}.en-fr.fr
		;;
	esac
	;;
"vi")
	testset=$testset_vi
	refdir=af-lib/iwslt15-envi-ytl/${testset}.vi
	;;
esac


# 1.2 select model
indir=results/models-v0en${LANGUAGE}/v0002-aaf-fr3.5-pretrain-lr0.001-seed4/${testset}

# 2.0 detok and bleu
FILE_TXT=translate-DETOK.txt
FILE_BLEU=bleu-DETOK.log
FILE_DIVERSE=diverse-DETOK.log

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


