# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import collections
import codecs
import numpy as np
import random

from utils.config import PAD, UNK, BOS, EOS, SPC

class Dataset(object):

	""" load src-tgt from file - does not involve torch """

	def __init__(self,
				# add params 
				path_src,
				path_tgt,
				path_vocab_src,
				path_vocab_tgt,
				attscore_path=None,
				max_seq_len=32,
				batch_size=64,
				use_gpu=True,
				use_type='word'
				):

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.path_vocab_src = path_vocab_src
		self.path_vocab_tgt = path_vocab_tgt
		self.attscore_path = attscore_path
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.use_type = use_type

		self.load_vocab()
		self.load_sentences()
		self.load_attscore()
		self.preprocess()

	def load_vocab(self):
		
		self.vocab_src = []
		self.vocab_tgt = []
		with codecs.open(self.path_vocab_src, encoding='UTF-8') as f:
			vocab_src_lines	= f.readlines()
		with codecs.open(self.path_vocab_tgt, encoding='UTF-8') as f:
			vocab_tgt_lines = f.readlines()

		self.src_word2id = collections.OrderedDict()
		self.tgt_word2id = collections.OrderedDict()
		self.src_id2word = collections.OrderedDict()
		self.tgt_id2word = collections.OrderedDict()

		def get_word_strip(w):
			return word.strip().split()[0] # remove \n, works in both cases
			# if self.use_type=='word': return word.strip() # remove \n
			# elif self.use_type=='char': return word.strip().split()[0] # remove \n
			# else: print('WARNING word not processed')

		for i, word in enumerate(vocab_src_lines):
			word = get_word_strip(word)
			self.vocab_src.append(word)
			self.src_word2id[word] = i
			self.src_id2word[i] = word

		for i, word in enumerate(vocab_tgt_lines):
			word = get_word_strip(word)
			self.vocab_tgt.append(word)
			self.tgt_word2id[word] = i
			self.tgt_id2word[i] = word


	def load_sentences(self):

		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()
		with codecs.open(self.path_tgt, encoding='UTF-8') as f:
			self.tgt_sentences = f.readlines()

		assert len(self.src_sentences) == len(self.tgt_sentences), 'Mismatch src:tgt - {}:{}' \
					.format(len(self.src_sentences),len(self.tgt_sentences))


	def load_attscore(self):

		""" laod reference attention scores """

		if self.attscore_path == None:
			self.attscore = None
		else:
			self.attscore = np.load(self.attscore_path)


	def preprocess(self):

		"""
			Use:
				map word2id once for all epoches (improved data loading efficiency)
				shuffling is done later
			Returns:
				0 - over the entire epoch
				1 - ids of src/tgt
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
			Note:
				src/tgt are always given
			Create
				self.train_src_word_ids
				self.train_src_sentence_lengths
				self.train_tgt_word_ids
				self.train_tgt_sentence_lengths
		"""

		self.vocab_size = {'src': len(self.src_word2id), 'tgt': len(self.tgt_word2id)}
		print("num_vocab_src: ", self.vocab_size['src'])
		print("num_vocab_tgt: ", self.vocab_size['tgt'])

		# declare temporary vars
		train_src_word_ids = []
		train_src_sentence_lengths = []
		train_tgt_word_ids = []
		train_tgt_sentence_lengths = []

		# import pdb #; pdb.set_trace()

		for idx in range(len(self.src_sentences)):
			src_sentence = self.src_sentences[idx]
			tgt_sentence = self.tgt_sentences[idx]
			src_words = src_sentence.strip().split()
			# tgt_words = tgt_sentence.strip().split()
			if self.use_type == 'char':
				tgt_words = tgt_sentence.strip()
			elif self.use_type == 'word':
				tgt_words = tgt_sentence.strip().split()

			# ignore long seq
			if len(src_words) > self.max_seq_len - 1 or len(tgt_words) > self.max_seq_len - 2:
				# src + EOS
				# tgt + BOS + EOS
				continue

			# print(src_sentence, src_words)
			# print(tgt_sentence, tgt_words)
			# pdb.set_trace()

			# source
			src_ids = [PAD] * self.max_seq_len
			for i, word in enumerate(src_words):
				if word==' ':
					assert self.use_type=='char'
					src_ids[i] = SPC
				elif word in self.src_word2id:
					src_ids[i] = self.src_word2id[word]
				else:
					src_ids[i] = UNK
			src_ids[i+1] = EOS
			train_src_word_ids.append(src_ids)
			train_src_sentence_lengths.append(len(src_words)+1) # include one EOS

			# target
			tgt_ids = [PAD] * self.max_seq_len
			tgt_ids[0] = BOS
			for i, word in enumerate(tgt_words):
				if word==' ':
					assert self.use_type=='char'
					tgt_ids[i+1] = SPC
				elif word in self.tgt_word2id:
					tgt_ids[i+1] = self.tgt_word2id[word]
				else:
					tgt_ids[i+1] = UNK
			tgt_ids[i+2] = EOS
			train_tgt_word_ids.append(tgt_ids)
			train_tgt_sentence_lengths.append(len(tgt_words)+2) # include EOS + BOS

			# print(src_ids)
			# print(tgt_ids)
			# pdb.set_trace()

		assert (len(train_src_word_ids) == len(train_tgt_word_ids)), "train_src_word_ids != train_tgt_word_ids"
		self.num_training_sentences = len(train_src_word_ids)
		print("num_sentences: ", self.num_training_sentences) # only those that are not too long

		# set class var to be used in batchify
		self.train_src_word_ids = train_src_word_ids
		self.train_src_sentence_lengths = train_src_sentence_lengths
		self.train_tgt_word_ids = train_tgt_word_ids
		self.train_tgt_sentence_lengths = train_tgt_sentence_lengths


	def construct_batches(self, is_train=False):

		"""
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 	a cat cat sat on the mat EOS PAD PAD ...
				tgt:	BOS a cat sat on the mat EOS PAD PAD ...
		"""

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids, self.train_src_sentence_lengths, self.train_tgt_sentence_lengths))
		if is_train:
			random.shuffle(_x)
		train_src_word_ids, train_tgt_word_ids, train_src_sentence_lengths, train_tgt_sentence_lengths = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end]}
			batches.append(batch)

		# add the last batch 
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths}
			batches.append(batch)

		print("num_batches: ", len(batches))

		return batches, self.vocab_size


	def construct_batches_with_attscore(self, is_train=False):

		"""
			Add reference att scores to each batch; 
			1. assuming seq > max_seqlen has been filtered out for both attscore & sentence ids
			2. truncating down last batch only happens in batchify & is_train=True
			
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
				attscore:	31 * 32 array 
		"""

		# same length for attscore & word ids 
		# both removed sentences that are too long
		# number of sentences not reduced before batchify!
		self.train_attscores = list(self.attscore)
		assert len(self.train_attscores) == len(self.train_src_word_ids), \
				'mismatch #sent in att:src {}:{}'.format(len(self.train_attscores), len(self.train_src_word_ids))

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids,
				self.train_src_sentence_lengths, self.train_tgt_sentence_lengths, self.train_attscores))
		if is_train:
			random.shuffle(_x)
		train_src_word_ids, train_tgt_word_ids, \
			train_src_sentence_lengths, train_tgt_sentence_lengths, train_attscores = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'attscores': train_attscores[i_start:i_end]}
			batches.append(batch)

		# add the last batch (underfull batch - add paddings)
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			dummy_prob = np.zeros((self.max_seq_len-1, self.max_seq_len))
			# dummy_prob.astype(np.double)
			# print(dummy_prob)
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []
			last_attscores = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_attscores.extend(train_attscores[i_start:i_end])
			last_attscores.extend([dummy_prob] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths,
				'attscores': last_attscores}
			batches.append(batch)

		print("num_batches: ", len(batches))

		return batches, self.vocab_size


	def construct_batches_with_attscore_sort(self, is_train=False, flag_sort=True):

		"""
			Add reference att scores to each batch; 
			1. assuming seq > max_seqlen has been filtered out for both attscore & sentence ids
			2. truncating down last batch only happens in batchify & is_train=True
			
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src: 			a  cat cat sat on the mat EOS PAD PAD ...
				tgt:		BOS a  cat sat on the mat EOS PAD PAD PAD ...
				attscore:	31 * 32 array 
		"""

		# same length for attscore & word ids 
		# both removed sentences that are too long
		# number of sentences not reduced before batchify!
		self.train_attscores = list(self.attscore)
		assert len(self.train_attscores) == len(self.train_src_word_ids), \
				'mismatch #sent in att:src {}:{}'.format(len(self.train_attscores), len(self.train_src_word_ids))

		# shuffle
		_x = list(zip(self.train_src_word_ids, self.train_tgt_word_ids,
				self.train_src_sentence_lengths, self.train_tgt_sentence_lengths, self.train_attscores))

		# import pdb; pdb.set_trace()
		if flag_sort:
			_x.sort(key = lambda x: x[3])
		elif is_train:
			random.shuffle(_x)
		# pdb.set_trace()

		train_src_word_ids, train_tgt_word_ids, \
			train_src_sentence_lengths, train_tgt_sentence_lengths, train_attscores = zip(*_x)

		batches = []

		for i in range(int(self.num_training_sentences/self.batch_size)):
			i_start = i * self.batch_size
			i_end = i_start + self.batch_size
			batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
				'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'attscores': train_attscores[i_start:i_end]}
			batches.append(batch)

		# add the last batch (underfull batch - add paddings)
		if not is_train and self.batch_size * len(batches) < self.num_training_sentences:
			dummy_id = [PAD] * self.max_seq_len
			dummy_length = 0
			dummy_prob = np.zeros((self.max_seq_len-1, self.max_seq_len))
			# dummy_prob.astype(np.double)
			# print(dummy_prob)
			
			i_start = self.batch_size * len(batches)
			i_end = self.num_training_sentences
			pad_i_start = i_end
			pad_i_end = i_start + self.batch_size
			
			last_src_word_ids = []
			last_tgt_word_ids = []
			last_src_sentence_lengths = []
			last_tgt_sentence_lengths = []
			last_attscores = []

			last_src_word_ids.extend(train_src_word_ids[i_start:i_end])
			last_src_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_tgt_word_ids.extend(train_tgt_word_ids[i_start:i_end])
			last_tgt_word_ids.extend([dummy_id] * (pad_i_end - pad_i_start))
			last_src_sentence_lengths.extend(train_src_sentence_lengths[i_start:i_end])
			last_src_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_tgt_sentence_lengths.extend(train_tgt_sentence_lengths[i_start:i_end])
			last_tgt_sentence_lengths.extend([dummy_length] * (pad_i_end - pad_i_start))
			last_attscores.extend(train_attscores[i_start:i_end])
			last_attscores.extend([dummy_prob] * (pad_i_end - pad_i_start))

			batch = {'src_word_ids': last_src_word_ids,
				'tgt_word_ids': last_tgt_word_ids,
				'src_sentence_lengths': last_src_sentence_lengths,
				'tgt_sentence_lengths': last_tgt_sentence_lengths,
				'attscores': last_attscores}
			batches.append(batch)

		random.shuffle(batches)
		print("num_batches: ", len(batches))

		return batches, self.vocab_size


def load_pretrained_embedding(word2id, embedding_matrix, embedding_path):

	""" assign value to src_word_embeddings and tgt_word_embeddings """

	counter = 0
	with codecs.open(embedding_path, encoding="UTF-8") as f:
		for line in f:
			items = line.strip().split()
			if len(items) <= 2:
				continue
			word = items[0].lower()
			if word in word2id:
				id = word2id[word]
				vector = np.array(items[1:])
				embedding_matrix[id] = vector
				counter += 1	

	print('loaded pre-trained embedding:', embedding_path)
	print('embedding vectors found:', counter)

	return embedding_matrix





























