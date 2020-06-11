#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt
BPE_TOKENS=40000

if [ ! -d "$SCRIPTS" ]; then
	echo "Please set SCRIPTS variable correctly to point to Moses scripts."
	exit
fi

src=en
tgt=fr
lang=en-fr
prep=iwslt15_en_fr

mkdir -p $prep
echo "pre-processing train data..."
for l in $src $tgt; do
	rm $prep/train.tags.$lang.$l
	sed '/^</ d' train.tags.$lang.$l | \
		perl $NORM_PUNC $l | \
		perl $REM_NON_PRINT_CHAR | \
		perl $TOKENIZER -threads 8 -a -l $l >> $prep/train.tags.$lang.$l
done

set1=dev2010
set2=tst2010
set3=tst2011
set4=tst2012
set5=tst2013

echo "pre-processing test data..."
for t in $set1 $set2 $set3 $set4 $set5; do 
	for l in $src $tgt; do
		rm -f $prep/IWSLT15.TED.$t.$lang.$l
		grep '<seg id' IWSLT15.TED.$t.$lang.$l.xml | \
			sed -e 's/<seg id="[0-9]*">\s*//g' | \
			sed -e 's/\s*<\/seg>\s*//g' | \
			sed -e "s/\’/\'/g" | \
		perl $TOKENIZER -threads 8 -a -l $l > $prep/IWSLT15.TED.$t.$lang.$l
		echo ""
	done
done

