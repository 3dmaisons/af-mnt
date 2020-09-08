#!/bin/bash

# Batch evaluate bleu score

EXP_DIR=/home/dawna/tts/qd212/models/af
cd $EXP_DIR

# 1.0 setup tools
SCRIPTS=/home/dawna/tts/qd212/models/af/af-lib/mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
BLEU_DETOK=$SCRIPTS/generic/multi-bleu-detok.perl

# 1.05 select mode
MODE=bleu # bleu diverse_train diverse_translate

# 1.1 select tgt language and testset
LANGUAGE=vi # fr de vi
testset_fr=tst2013 # tst2013 tst2014
testset_de=tst-COMMON
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
"de")
	testset=$testset_de
	refdir=af-lib/tst-COMMON.de
	# refdir=af-lib/mustc-en-de/tst-COMMON/tst-COMMON.de
	;;
"vi")
	testset=$testset_vi
	refdir=af-lib/iwslt15-envi-ytl/${testset}.vi
	;;
esac



# 1.2 select model
# indir=af-models/tf/${testset}

# indir=results/models-v9enfr/tf-v0001/${testset}
	# indir=results/models-v9enfr/af-v0001/${testset}
	# indir=results/models-v9enfr/afdynamic-v0001/${testset}
	# indir=results/models-v9enfr/sched-af-v0001/${testset}

# indir=results/models-v9enfr/aaf-v0001-tf/${testset}
	# indir=results/models-v9enfr/aaf-v0002-tf/${testset}
	# indir=results/models-v9enfr/aaf-v0002-tf-bs50-v2/${testset}
	# indir=results/models-v9enfr/aaf-v0002-tf-bs50-pretrain/${testset}
	# indir=results/models-v9enfr/aaf-v0002-tf-bs50-pretrain-asup/${testset}
	# indir=results/models-v9enfr/aaf-v0002-tf-bs50-pretrain-lr0.001/${testset}
	# indir=results/models-v9enfr/aaf-v0002-tf-bs50-pretrain-lr0.001-seed6/${testset}
	# indir=results/models-v9enfr/aaf-v0004-ss/${testset}
	# indir=results/models-v9enfr/aaf-v0004-ss-max0.2linear/${testset}
	# indir=results/models-v9enfr/aaf-v0004-ss-max0.2linear-asup-check/${testset}

# indir=results/models-v9enfr/aaf-v0010-af/${testset}
	# indir=results/models-v9enfr/aaf-v0013-af/${testset}
	# indir=results/models-v9enfr/aaf-v0013-af-pretrain/${testset}
	# indir=results/models-v9enfr/aaf-v0013-af-fixkl/${testset}
	# indir=results/models-v9enfr/aaf-v0013-aftf/${testset}
	# indir=results/models-v9enfr/aaf-v0013-aftf-bs128/${testset}
	# indir=results/models-v9enfr/aaf-v0013-aftf-pretrain/${testset}
	# indir=results/models-v9enfr/aaf-v0013-aftf-pretrain-lr0.001/${testset}

# indir=results/models-v9enfr/aaf-v0020-sched-debug/${testset}
	# indir=results/models-v9enfr/aaf-v0020-sched-log/${testset}
	# indir=results/models-v9enfr/aaf-v0020-sched-fr15/${testset}
	# indir=results/models-v9enfr/aaf-v0020-sched-fr0.0/${testset}
	# indir=results/models-v9enfr/aaf-v0020-sched-fr2.5/${testset}
	# indir=results/models-v9enfr/aaf-v0020-sched-fr3.25/${testset}
	# indir=results/models-v9enfr/aaf-v0020-sched-fr4.0/${testset}

# indir=results/models-v9enfr/aaf-v0030-sched-fr2.0-pretrain/${testset}
	# indir=results/models-v9enfr/aaf-v0030-sched-fr3.0-pretrain-v2/${testset}
	# indir=results/models-v9enfr/aaf-v0030-sched-fr3.1-pretrain/${testset}
	# indir=results/models-v9enfr/aaf-v0030-sched-fr3.0-pretrain-lr0.001/${testset}
	# indir=results/models-v9enfr/aaf-v0030-sched-fr2.5-pretrain-lr0.001-smoothKL/${testset}
	# indir=results/models-v9enfr/aaf-v0030-sched-fr3.5-pretrain-lr0.001-smoothKL-seed8/${testset}

# indir=results/models-v9enfr/aaf-v0040-paf-nbFrToken20/${testset}
	# indir=results/models-v9enfr/aaf-v0040-paf/${testset}
	# indir=results/models-v9enfr/aaf-v0040-paf-nbFrToken5-lr0.001/${testset}

# indir=results/models-v9enfr/aaf-v0050-oaf/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-asup-detach/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-asup-sched/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-asup/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-notf/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-sched/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-gamma50/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf1.0/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf0.5-alwaysKL/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf0.8-noKL/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf0.8-alwaysKLsmooth1.0/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf0.9-alwaysKLsmooth0.5/${testset}
	# indir=results/models-v9enfr/aaf-v0050-oaf-tf0.8-alwaysMSE20.0/${testset}

# indir=results/models-v9enfr/aaf-v0050-aoaf-tf0.9-selectKL2.0/${testset}

# indir=results/models-v0ende/v0000-tf-lr0.001/${testset}

# indir=results/models-v0en${LANGUAGE}/v0000-tf-lr0.002/${testset}
# indir=results/models-v0en${LANGUAGE}/v0001-af-lr0.002/${testset}
indir=results/models-v0en${LANGUAGE}/v0001-af-pretrain-lr0.001/${testset}
# indir=results/models-v0en${LANGUAGE}/v0000-tf-pretrain-lr0.001-seed2/${testset}
# indir=results/models-v0en${LANGUAGE}/v0001-aftf-lr0.002/${testset}
# indir=results/models-v0en${LANGUAGE}/v0002-aaf-fr3.5-pretrain-lr0.001-seed4/${testset}

# d_arr=(2 4 6 8)
# ep_arr=(8 7 16 9 21)
# tmp_dir=results/models-v9enfr/aaf-v0002-tf-bs50-pretrain-lr0.001
# ep_arr=(15 9 24 15 7)
# tmp_dir=results/models-v9enfr/aaf-v0030-sched-fr3.0-pretrain-lr0.001-smoothKL

d_arr=(8)
ep_arr=(24 29)
tmp_dir=results/models-v9enfr/aaf-v0030-sched-fr3.5-pretrain-lr0.001-smoothKL

# ep_arr=(22 9 24 30 12)
# tmp_dir=results/models-v0en${LANGUAGE}/v0000-tf-pretrain-lr0.001
# ep_arr=(26 22 17 30 22)
# tmp_dir=results/models-v0en${LANGUAGE}/v0002-aaf-fr3.5-pretrain-lr0.001

for i in ${!d_arr[@]}; do
	d_arr[$i]=${EXP_DIR}/${tmp_dir}-seed${d_arr[${i}]}/${testset}/epoch_${ep_arr[${i}]}
done
if [ -d ${tmp_dir}/${testset} ]; then
	d_arr+=(${EXP_DIR}/${tmp_dir}/${testset}/epoch_${ep_arr[-1]})
else
	d_arr+=(${EXP_DIR}/${tmp_dir}-seed16/${testset}/epoch_${ep_arr[-1]})
fi


# 2.0 detok and bleu
FILE_TXT=translate-DETOK.txt
FILE_BLEU=bleu-DETOK.log
FILE_DIVERSE=diverse-DETOK.log

trap "exit" INT
case $MODE in
"bleu")
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
;;
"diverse_train")
	for d_gen in ${d_arr[@]}; do
		echo gen $d_gen > ${d_gen}/${FILE_DIVERSE}
		for d_ref in ${d_arr[@]}; do
			if [ ! -f ${d_gen}/${FILE_TXT} ]; then
			echo detok, saving to ${d_gen}/${FILE_TXT}
			perl ${DETOKENIZER} -l ${LANGUAGE} < ${d_gen}/translate.txt > ${d_gen}/translate-DETOK.txt
			fi
			if [ ! -f ${d_ref}/${FILE_TXT} ]; then
			echo detok, saving to ${d_ref}/${FILE_TXT}
			perl ${DETOKENIZER} -l ${LANGUAGE} < ${d_ref}/translate.txt > ${d_ref}/translate-DETOK.txt
			fi
			# if [ ! -f ${d_gen}/${FILE_DIVERSE} ]; then
			echo ref $d_ref >> ${d_gen}/${FILE_DIVERSE}
			echo BLEU score, saving to ${d_gen}/${FILE_DIVERSE}
			perl ${BLEU_DETOK} ${d_ref}/${FILE_TXT} < ${d_gen}/${FILE_TXT} >> ${d_gen}/${FILE_DIVERSE}
			# fi
		done
	done
;;
"diverse_translate")
	FILE_DIVERSE=diverse-translate-DETOK.log
	for d_gen in ${d_arr[@]}; do
		for i in {0..4}; do
			FILE_TXT=translate-run${i}-DETOK.txt
			if [ ! -f ${d_gen}/${FILE_TXT} ]; then
			echo detok, saving to ${d_gen}/${FILE_TXT}
			perl ${DETOKENIZER} -l ${LANGUAGE} < ${d_gen}/translate-run${i}.txt > ${d_gen}/${FILE_TXT}
			fi
		done
		for i_gen in {0..4}; do
			FILE_TXT_GEN=translate-run${i_gen}-DETOK.txt
			for i_ref in {0..4}; do
				FILE_TXT_REF=translate-run${i_ref}-DETOK.txt
				echo BLEU score, saving to ${d_gen}/${FILE_DIVERSE}
				perl ${BLEU_DETOK} ${d_gen}/${FILE_TXT_REF} < ${d_gen}/${FILE_TXT_GEN} >> ${d_gen}/${FILE_DIVERSE}
			done
		done
	done
;;
esac


# previous version of BLEU scorer, taking tokenized txt as input

# FILE_TXT=translate.txt
# FILE_BLEU=bleu.log
# for d in ${EXP_DIR}/${indir}/*; do
# 	if [ ! -f ${d}/${FILE_BLEU} ]; then
# 	echo saving to $d/${FILE_BLEU}
# 	python /home/dawna/tts/qd212/models/af/bleu_scorer.py $d/${FILE_TXT} $refdir > $d/${FILE_BLEU}
# 	fi
# done