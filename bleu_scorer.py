#!/usr/bin/python
'''
	bleu scorer
'''

import sys
import os
import numpy as np
import datetime
import argparse
import collections
import nltk

import warnings
warnings.filterwarnings("ignore")

def main(arguments):

	# read in command line args
	srcfname = arguments.srcfname
	srcfname2 = arguments.srcfname2


	# write out commands
	# ------------------
	hyppath = sys.argv[1]
	gldpath = sys.argv[2]

	hypf = open(hyppath, 'r')
	hyplines = hypf.readlines()
	hypf.close()
        gldf = open(gldpath, 'r')
        gldlines = gldf.readlines()
        gldf.close()
	
	assert len(hyplines) == len(gldlines), 'hyp {} : ref {}'.format(len(hyplines), len(gldlines))

	hyps = []
	glds = []
	score = 0 
	n = len(hyplines)
	for idx in range(len(hyplines)):
		hyp = hyplines[idx].strip().split()
		gld = gldlines[idx].strip().split()
		#print(hyp)
		#print(gld)
		hyps.append(hyp)
		glds.append([gld])
		score += nltk.translate.bleu_score.sentence_bleu([gld], hyp, weights = (0.5, 0.5))
		#print(score)
		#raw_input('...')
		#print(idx,n)		

	score = 1. * score / n
	print('avg sentence bigram BLEU')
	print(score)

	corpus_score_a = nltk.translate.bleu_score.corpus_bleu(glds, hyps, weights = (0.5, 0.5))
	corpus_score_b = nltk.translate.bleu_score.corpus_bleu(glds, hyps, weights = (0.25, 0.25, 0.25, 0.25))
	print('bigram corpus BLEU')
	print(corpus_score_a)
        print('4gram corpus BLEU')
        print(corpus_score_b)



if __name__ == '__main__':
    print(__doc__)
    commandLineParser = argparse.ArgumentParser (
        description = 'txt to tsv format')

    commandLineParser.add_argument ('srcfname',
        metavar = 'srcfname', type = str,
        help = 'hyp')

    commandLineParser.add_argument ('srcfname2',
        metavar = 'srcfname2', type = str,
        help = 'ref')

    arguments = commandLineParser.parse_args()
    main(arguments)









