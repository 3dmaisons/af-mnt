import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

sys.path.append('/home/mifs/ytl28/af/af-scripts/')
from utils.dataset import Dataset
from utils.misc import set_global_seeds, print_config, save_config, validate_config, get_memory_alloc
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor, plot_alignment
from utils.config import PAD, EOS
from modules.loss import NLLLoss, KLDivLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import Seq2Seq_DD

import logging
logging.basicConfig(level=logging.INFO)

device = 'cpu'
KEY_ATTN_REF = 'attention_ref'
KEY_ATTN_SCORE = 'attention_score'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'
KEY_MODEL_STRUCT = 'model_struct'


def load_arguments(parser):

	""" Seq2Seq-DD eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, required=True, help='test tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')	
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_tf', type=str, default=None, help='model load tf dir')
	parser.add_argument('--load_af', type=str, default=None, help='model load af dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')
	parser.add_argument('--test_attscore_path', type=str, default=None, help='test set attention score reference')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')	
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')	
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--use_teacher', type=str, default='False', help='whether or not feeding reference sequence')
	parser.add_argument('--mode', type=int, default=2, help='which mode to run on')
	parser.add_argument('--gen_mode', type=str, default='afdynamic', help='tf af generation mode\
											afdynamic: tf: y_a; af: a_t \
											afstatic: tf: y; af: a_t ')

	return parser

def translate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width):

	""" 
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)	
	model.reset_batch_size(test_set.batch_size)	
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	model.set_var('debug_count', 0)	
	model.to(device)

	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	# f = open(os.path.join(test_path_out, 'translate.txt'), 'w') -> use proper encoding
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		match = 0
		total = 0
		with torch.no_grad():
			for batch in test_batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)			

				src_ids = _convert_to_tensor(src_ids, use_gpu)
				decoder_outputs, decoder_hidden, other = model(src=src_ids, 
																is_training=False,
																att_key_feats=src_probs, 
																beam_width=beam_width)
				# memory usage
				mem_kb, mem_mb, mem_gb = get_memory_alloc()
				mem_mb = round(mem_mb, 2)
				print('Memory used: {0:.2f} MB'.format(mem_mb))
				
				# write to file
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					# skip padding sentences in batch (num_sent % batch_size != 0)
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						outline = ' '.join(words)
					f.write('{}\n'.format(outline))
					# if i == 0: 
					# 	print(outline)
				sys.stdout.flush()


def evaluate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width):

	""" 
		with reference tgt given - Run translation.
		Args:
			test_set: test dataset
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
		Returns:
			accuracy (excluding PAD tokens)
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))	
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)	
	model.reset_batch_size(test_set.batch_size)	
	model.set_beam_width(beam_width)	
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	model.to(device)

	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	# f = open(os.path.join(test_path_out, 'test.txt'), 'w')
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		match = 0
		total = 0
		with torch.no_grad():
			for batch in test_batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

				decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids, 
																is_training=False,
																att_key_feats=src_probs,
																beam_width=beam_width)

				# Evaluation
				seqlist = other['sequence'] # traverse over time not batch
				if beam_width > 1: 
					full_seqlist = other['topk_sequence']
					decoder_outputs = decoder_outputs[:-1]
				for step, step_output in enumerate(decoder_outputs):
					target = tgt_ids[:, step+1]
					non_padding = target.ne(PAD)
					correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()

				# write to file
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					# skip padding sentences in batch (num_sent % batch_size != 0)
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						outline = ' '.join(words)
					f.write('{}\n'.format(outline))
					if i == 0: 
						print(outline)
				sys.stdout.flush()

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		
	return accuracy


def debug(test_set, load_dir, use_gpu, max_seq_len, beam_width):

	""" 
		debug - print out weights / attention.
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu			
		Returns:
			
	"""

	# check devide
	print('cuda available: {}'.format(torch.cuda.is_available()))
	use_gpu = use_gpu and torch.cuda.is_available()

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# DEBUG
	for name, param in model.named_parameters():
		# if 'embedder' in name:
		# 	print('{}:{}'.format(name, param[5]))
		if 'att.linear_att_ak.weight' in name:
			print('{}:{},{}'.format(name, param.size(), param[:,-3:]))
			input('...')
		if 'att.linear_att_bk.weight' in name:
			print('{}:{},{}'.format(name, param.size(), param[:,-3:]))
			input('...')
		if 'att.linear_att_ck.weight' in name:
			print('{}:{},{}'.format(name, param.size(), param[:,-3:]))
			input('...')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)
	model.reset_batch_size(test_set.batch_size)
	
	# note in debug mode: always turn off beam search to print att score 
	model.set_beam_width(beam_width=0)

	print('max seq len {}'.format(model.max_seq_len))
	print('max seq len {}'.format(max_seq_len))
		
	# load test
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	model.eval()
	match = 0
	total = 0
	with torch.no_grad():
		for batch in test_batches:

			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

			decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids, 
															is_training=False,
															att_key_feats=src_probs,
															beam_width=0)

			decoder_outputs_pseudo, decoder_hidden_pseudo, other_pseudo = model(src_ids, tgt_ids, 
															is_training=True, 
															att_key_feats=src_probs,
															beam_width=0) # teacher forcing 

			# Evaluation
			attention = other['attention_score'] # 64 x 31 x 32 (batch_size x tgt_len(query_len) x src_len(key_len))
			seqlist = other['sequence'] # traverse over time not batch
			bsize = test_set.batch_size
			max_seq = test_set.max_seq_len
			vocab_size = len(test_set.tgt_word2id)
			logp = np.empty([bsize, max_seq]) 			# logp for hyp seq | conditioned on hyp seq
			logp_pseudo = np.empty([bsize, max_seq]) 	# logp for ref seq | conditioned on ref seq
			logp_cross = np.empty([bsize, max_seq]) 	# logp for ref curr word | conditioned on hyp seq
			# logp_debug = np.empty([bsize, max_seq]) 	# 
			for idx in range(len(decoder_outputs)): # loop over max_seq
				step = idx
				step_output = decoder_outputs[idx] # 64 x vocab_size
				step_output_pseudo = decoder_outputs_pseudo[idx]
				# count correct
				target = tgt_ids[:, step+1]
				non_padding = target.ne(PAD)
				correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
				match += correct
				total += non_padding.sum().item()
				# collect logp
				ids = seqlist[step] # hyp ids
				ids_pseudo = target # ref idx
				for j in range(bsize):
					logp[j, idx] = '{:.2e}'.format(float(step_output[j][ids[j]]))
					logp_pseudo[j, idx] = '{:.2e}'.format(float(step_output_pseudo[j][ids_pseudo[j]]))
					logp_cross[j, idx] = '{:.2e}'.format(float(step_output[j][ids_pseudo[j]]))
					# logp_debug[j, idx] = '{:.2e}'.format(float(max(step_output[j])))
					# print('logp[{}, {}] {}'.format(j, idx, logp[j, idx]))
					# print('logp_pseudo[{}, {}] {}'.format(j, idx, logp_pseudo[j, idx]))
					# print('logp_cross[{}, {}] {}'.format(j, idx, logp_cross[j, idx]))
					# print('logp_debug[{}, {}] {}'.format(j, idx, logp_debug[j, idx]))
					# input('...')

			# Print sentence by sentence
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			for i in range(len(seqwords)): # loop over sentences
				outline_src = 'SRC: {}\n'.format(' '.join(srcwords[i]))
				outline_ref = 'REF: {}\n'.format(' '.join(refwords[i]))
				outline_gen = 'GEN: {}\n'.format(' '.join(seqwords[i]))
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline)
				# summing logp
				loc_eos = seqwords[i].index('</s>')
				loc_eos_pseudo = refwords[i].index('</s>')
				sum_logp = sum(logp[i, :loc_eos])
				sum_logp_pseudo = sum(logp_pseudo[i, :loc_eos_pseudo])
				sum_logp_cross = sum(logp_cross[i, :loc_eos_pseudo])
				# sum_logp_debug = sum(logp_debug[i, :loc_eos_pseudo])
				print('logp GEN: {:.2e}, {}'.format(sum_logp, logp[i, :loc_eos]))
				print('logp REF: {:.2e}, {}'.format(sum_logp_pseudo, logp_pseudo[i, :loc_eos_pseudo]))
				print('logp REF conditioned on GEN: {:.2e}, {}'.format(sum_logp_cross, logp_cross[i, :loc_eos_pseudo]))
				# print('logp GEN MAX: {:.2e}, {}'.format(sum_logp_debug, logp_debug[i, :loc_eos]))
				for j in range(len(attention)):
					# i: idx of batch
					# j: idx of query
					gen = seqwords[i][j]
					ref = refwords[i][j]
					print('REF:GEN - {}:{}'.format(ref,gen))
					print('{}th ATT size: {}'.format(j, attention[j][i].size()))
					att = attention[j][i]
					print(att)
					print(torch.argmax(att))
					print(sum(sum(att)))
					input('Press enter to continue ...')

	if total == 0:
		accuracy = float('nan')
	else:
		accuracy = match / total
	print(accuracy)


def debug_beam_search(test_set, load_dir, use_gpu, max_seq_len, beam_width):

	""" 
		with reference tgt given - debug beam search.
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
		Returns:
			accuracy (excluding PAD tokens)
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))	
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)	
	model.reset_batch_size(test_set.batch_size)	
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	model.eval()
	match = 0
	total = 0
	with torch.no_grad():
		for batch in test_batches:

			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

			decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids, 
															is_training=False,
															att_key_feats=src_probs,
															beam_width=beam_width)

			# Evaluation
			seqlist = other['sequence'] # traverse over time not batch
			if beam_width > 1: 
				# print('dict:sequence')
				# print(len(seqlist))
				# print(seqlist[0].size())

				full_seqlist = other['topk_sequence']
				# print('dict:topk_sequence')
				# print(len(full_seqlist))
				# print((full_seqlist[0]).size())
				# input('...')
				seqlists = []
				for i in range(beam_width):
					seqlists.append([seq[:, i] for seq in full_seqlist])

				# print(decoder_outputs[0].size())
				# print('tgt id size {}'.format(tgt_ids.size()))
				# input('...')

				decoder_outputs = decoder_outputs[:-1]
				# print(len(decoder_outputs))

			for step, step_output in enumerate(decoder_outputs): # loop over time steps
				target = tgt_ids[:, step+1]
				non_padding = target.ne(PAD)
				# print('step', step)
				# print('target', target)
				# print('hyp', seqlist[step])
				# if beam_width > 1: 
				# 	print('full_seqlist', full_seqlist[step])
				# input('...')
				correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
				match += correct
				total += non_padding.sum().item()

			# write to file
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)			
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			seqwords_list = []
			for i in range(beam_width):
				seqwords_list.append(_convert_to_words(seqlists[i], test_set.tgt_id2word))

				outline_src = 'SRC: {}\n'.format(' '.join(srcwords[i]))
				outline_ref = 'REF: {}\n'.format(' '.join(refwords[i]))
				outline_gen = 'GEN: {}\n'.format(' '.join(seqwords[i]))
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline)

			for i in range(len(seqwords)):
				outline_ref = ' '.join(refwords[i]) 
				outline_ref = 'REF: {}\n'.format(outline_ref)
				sys.stdout.buffer.write(outline_ref)		
				outline_hyp = ' '.join(seqwords[i]) 
				# print(outline_hyp)
				outline_hyps = []
				for j in range(beam_width):
					outline_hyps.append(' '.join(seqwords_list[j][i]))
					print('{}th'.format(j), outline_hyps[-1])

				# skip padding sentences in batch (num_sent % batch_size != 0)
				# if src_lengths[i] == 0:
				# 	continue
				# words = []
				# for word in seqwords[i]:
				# 	if word == '<pad>':
				# 		continue
				# 	elif word == '</s>':
				# 		break
				# 	else:
				# 		words.append(word)
				# if len(words) == 0:
				# 	outline = ''
				# else:
				# 	outline = ' '.join(words)

				input('...')

			sys.stdout.flush()

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		
	return accuracy


def translate_att_forcing(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, use_teacher):

	""" 
		MODE = 7
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)	
	model.reset_batch_size(test_set.batch_size)	
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	model.to(device)

	model.set_var('attention_forcing', False)
	# load test
	if type(test_set.attscore_path) != type(None):
		test_batches, vocab_size = test_set.construct_batches_with_attscore(is_train=False)
		model.set_var('attention_forcing', True)
	else:
		test_batches, vocab_size = test_set.construct_batches(is_train=False)

	print('max seq len {}'.format(model.max_seq_len))
	print('attention_forcing {}'.format(model.attention_forcing))
	sys.stdout.flush()
	prefix = ''
	
	if use_teacher:
		teacher_forcing_ratio = 1.0 # teacher forcing or not 
		prefix += 'tgt' # or ''
	else:
		teacher_forcing_ratio = 0.0

	# ===================================================================[config] 
	# ----- can be changed ---------- 
	is_training = True # or False
	# ------------------------------- 
	if is_training == False:
		teacher_forcing_ratio=0.0
		prefix = ''

	# record config
	f = open(os.path.join(test_path_out, 'conf.log'), 'w')
	f.write('Model dir: {}\n'.format(latest_checkpoint_path))
	f.write('is training: {}\n'.format(is_training))
	f.write('teacher forcing mode: {}\n'.format(use_teacher))
	f.write('attention forcing mode: {}\n'.format(model.attention_forcing))
	f.write('max_seq_len: {}\n'.format(model.max_seq_len))
	f.write('mode: {}\n'.format(prefix))
	f.close()
	# ===================================================================


	# f = open(os.path.join(test_path_out, 'translate.txt'), 'w') -> use proper encoding
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		match = 0
		total = 0
		with torch.no_grad():
			for bidx in range(len(test_batches)):
				batch = test_batches[bidx]
				print(bidx)
				# batch = test_batches[len(test_batches)-1-bidx]
				# print(len(test_batches)-1-bidx)

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']

				src_probs = None
				attscores = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)			
				if 'attscores' in batch:
					attscores_dummy = batch['attscores'] #list of numpy arrays
					attscores = _convert_to_tensor(attscores_dummy, use_gpu) # n*199*200
					# print(len(attscores_dummy))
					# print(attscores_dummy[-1].shape)
					# print(attscores_dummy[0])

				if is_training == False:
					attscores = None

				# print(attscores.size()) # 32 * 199 * 200
				# input('...')
				src_ids = _convert_to_tensor(src_ids, use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

				decoder_outputs, decoder_hidden, other = model(src=src_ids, tgt=tgt_ids,
																is_training=is_training, 
																teacher_forcing_ratio=teacher_forcing_ratio, # can be changed
																att_key_feats=src_probs, 
																att_scores=attscores,
																beam_width=beam_width)
				# memory usage
				mem_kb, mem_mb, mem_gb = get_memory_alloc()
				mem_mb = round(mem_mb, 2)
				print('Memory used: {0:.2f} MB'.format(mem_mb))
				
				# write to file
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					# skip padding sentences in batch (num_sent % batch_size != 0)
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						outline = ' '.join(words)
					f.write('{}\n'.format(outline))
					# if i == 0: 
					# 	print(outline)
				sys.stdout.flush()


def att_plot(test_set, load_dir, plot_path, use_gpu, max_seq_len, beam_width, use_teacher):

	""" 
		MODE = 4
		generate attention alignment plots
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
			max_seq_len
		Returns:
	"""

	# check devide
	print('cuda available: {}'.format(torch.cuda.is_available()))
	use_gpu = use_gpu and torch.cuda.is_available()

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)
	model.reset_batch_size(test_set.batch_size)
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	model.set_beam_width(beam_width=1)	# in plotting mode always turn off beam search
	model.set_var('debug_count', 0)

	# model.create_dummy_classvar(var_name)
	print('max seq len {}'.format(model.max_seq_len))
		
	# load test - att forcing or not
	model.set_var('attention_forcing', False)
	if type(test_set.attscore_path) != type(None):
		test_batches, vocab_size = test_set.construct_batches_with_attscore(is_train=False)
		model.set_var('attention_forcing', True)
		prefix = 'att'
	else:
		test_batches, vocab_size = test_set.construct_batches(is_train=False)
		prefix = ''		

	if use_teacher:
		teacher_forcing_ratio = 1.0 # teacher forcing or not 
		prefix += 'tgt' # or ''
	else:
		teacher_forcing_ratio = 0.0
		

	# ===================================================================[config] 
	# ----- can be changed ---------- 
	is_training = True # True or False
	# ------------------------------- 
	if is_training == False:
		teacher_forcing_ratio=0.0
		prefix = ''

	# record config
	f = open(os.path.join(plot_path, 'conf.log'), 'w')
	f.write('Model dir: {}\n'.format(latest_checkpoint_path))
	f.write('is training: {}\n'.format(is_training))
	f.write('teacher forcing mode: {}\n'.format(use_teacher))
	f.write('attention forcing mode: {}\n'.format(model.attention_forcing))
	f.write('max_seq_len: {}\n'.format(model.max_seq_len))
	f.write('mode: {}\n'.format(prefix))
	f.close()

	print('mode: {}'.format(prefix))
	# ===================================================================

	# start eval
	model.eval()
	match = 0
	total = 0
	count = 0
	count2 = 0
	with torch.no_grad():
		for batch in test_batches:

			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			attscores = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)
			if 'attscores' in batch:
				attscores_dummy = batch['attscores'] #list of numpy arrays
				attscores = _convert_to_tensor(attscores_dummy, use_gpu)

			if is_training == False:
				attscores = None

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

			decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids, 
															is_training=is_training, 
															teacher_forcing_ratio=teacher_forcing_ratio, 
															att_key_feats=src_probs,
															att_scores=attscores,
															beam_width=1)
			# Evaluation
			# default batch_size = 1
			# attention: 31 * [1 x 1 x 32] ( tgt_len(query_len) * [ batch_size x 1 x src_len(key_len)] )
			attention = other['attention_score'] 
			seqlist = other['sequence'] # traverse over time not batch
			bsize = test_set.batch_size
			max_seq = test_set.max_seq_len
			vocab_size = len(test_set.tgt_word2id)
			for idx in range(len(decoder_outputs)): # loop over max_seq
				step = idx
				step_output = decoder_outputs[idx] # 64 x vocab_size
				# count correct
				target = tgt_ids[:, step+1]
				non_padding = target.ne(PAD)
				correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
				match += correct
				total += non_padding.sum().item()

			# Print sentence by sentence
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			# print(type(attention))
			# print(len(attention))
			# print(type(attention[0]))
			# print(attention[0].size())
			# input('...')
			n_q = len(attention)
			n_k = attention[0].size(2)
			b_size =  attention[0].size(0)
			att_score = torch.empty(n_q, n_k, dtype=torch.float)
			# att_score = np.empty([n_q, n_k])
			
			for i in range(len(seqwords)): # loop over sentences
				outsrc = 'SRC: {}\n'.format(' '.join(srcwords[i])).encode('utf-8')
				outref = 'REF: {}\n'.format(' '.join(refwords[i])).encode('utf-8')
				outgen = 'GEN: {}\n'.format(' '.join(seqwords[i])).encode('utf-8')
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outgen)
				for j in range(len(attention)):
					# i: idx of batch
					# j: idx of query
					gen = seqwords[i][j]
					ref = refwords[i][j]
					att = attention[j][i]
					# print('REF:GEN - {}:{}'.format(ref,gen))
					# print('{}th ATT size: {}'.format(j, attention[j][i].size()))
					# print(att)
					# print(torch.argmax(att))
					# print(sum(sum(att)))

					# record att scores
					att_score[j] = att
					# input('Press enter to continue ...')

				# plotting 
				# print(att_score)
				if '</s>' in srcwords[i]:
					loc_eos_k = srcwords[i].index('</s>') + 1
				else:
					loc_eos_k = len(srcwords[i])
				if '</s>' in seqwords[i]:
					loc_eos_q = seqwords[i].index('</s>') + 1
				else:
					loc_eos_q = len(seqwords[i])
				if '</s>' in refwords[i]:
					loc_eos_ref = refwords[i].index('</s>') + 1
				else:
					loc_eos_ref = len(refwords[i])

				att_score_trim = att_score[:loc_eos_q, :loc_eos_k] # each row (each query) sum up to 1
				# print('eos_k: {}, eos_q: {}'.format(loc_eos_k, loc_eos_q))
				# print(att_score_trim)
				print('\n')

				choice1 = input('Save att score or not ? - y/n\n')
				if choice1:
					if choice1.lower()[0] == 'y':
						print('saving att ...')
						rec_dir = os.path.join(plot_path, '{}-{}.att.txt'.format(prefix,count))
						print(rec_dir)
						recf = open(rec_dir, 'w')
						att = att_score.data.cpu().numpy().tolist()
						# print(att)
						recf.write('SRC: {}\n'.format(outsrc))
						recf.write('REF: {}\n'.format(outref))
						recf.write('GEN: {}\n'.format(outgen))

						recf.write('{} x {}\n\n'.format(len(att),len(att[0])))
						for idx in range(len(att)):
							for j in range(len(att[0])):
								recf.write('{0:.2f} '.format(att[idx][j]))
							recf.write('\n')
						recf.close()
						count += 1
						input('Press enter to continue ...')
				
				choice = input('Plot or not ? - y/n\n')
				if choice:
					if choice.lower()[0] == 'y':
						print('plotting ...')
						plot_dir = os.path.join(plot_path, '{}-{}.png'.format(prefix,count2))
						print(plot_dir)
						src = srcwords[i][:loc_eos_k]
						hyp = seqwords[i][:loc_eos_q]
						ref = refwords[i][:loc_eos_ref]
						# x-axis: src; y-axis: hyp
						plot_alignment(att_score_trim.numpy(), plot_dir, src=src, hyp=hyp, ref=ref) 
						# plot_alignment(att_score_trim.numpy(), plot_dir, src=src, hyp=hyp, ref=None) # no ref 				
						count2 += 1
						input('Press enter to continue ...')


	if total == 0:
		accuracy = float('nan')
	else:
		accuracy = match / total
	print(accuracy)


def print_attscore(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, use_teacher):

	""" 
		MODE = 6
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)	
	model.reset_batch_size(test_set.batch_size)	
	model.check_classvar('attention_forcing')
	model.check_classvar('num_unilstm_enc')
	model.check_classvar('residual')
	model.set_var('debug_count', 0)
	model.set_var('attention_forcing', False) # in print att mode always False
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load trainset
	# remove too long sentences (no further #sent reduction before batchify)
	# note: final unfull batch ignored during batchify - after zipped with srcids
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	# ===================================================================[config] 
	# ----- can be changed ---------- 
	is_training = True # True or False:turn off both tf and af
	# ------------------------------- 
	if use_teacher:
		teacher_forcing_ratio = 1.0 # teacher forcing or not 
	else:
		teacher_forcing_ratio = 0.0

	# record config
	f = open(os.path.join(test_path_out, 'conf.log'), 'w')
	f.write('Model dir: {}\n'.format(latest_checkpoint_path))
	f.write('is training: {}\n'.format(is_training))
	f.write('teacher forcing mode: {}\n'.format(use_teacher))
	f.write('attention forcing mode: {}\n'.format(model.attention_forcing))
	f.write('max_seq_len: {}\n'.format(model.max_seq_len))
	# ===================================================================

	# f = open(os.path.join(test_path_out, 'translate.txt'), 'w') -> use proper encoding
	# with open(os.path.join(test_path_out, 'att_score.npy'), 'w', encoding="utf8") as f:
	model.eval()
	match = 0
	total = 0
	att_scores_lis = []
	with torch.no_grad():
		for idx in range(len(test_batches)):

			batch = test_batches[idx]
			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)			

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

			decoder_outputs, decoder_hidden, other = model(src=src_ids, tgt=tgt_ids,
															is_training=is_training,
															teacher_forcing_ratio=teacher_forcing_ratio,
															att_key_feats=src_probs,
															beam_width=beam_width)
			# memory usage
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			if idx % 500 == 0:
				print('Memory used: {0:.2f} MB'.format(mem_mb))
			sys.stdout.flush()

			# write to file
			seqlist = other['sequence']
			att_score = other['attention_score']
			att_score_lis = []
			# print('time steps', len(att_score))
			# print('att scores size', att_score[0].size())
			for i in range(len(att_score)):
				# print(att_score[i][0][0].numpy())
				# print(len(att_score)) # 63
				# print(att_score[i].size()) # b * 1 * 64 
				# input('...')
				# att_score_lis.append(att_score[i][0][0].cpu().numpy())
				att_score_morph = att_score[i].view(-1, att_score[-1].size(-1))  # b * 64 
				# print(att_score_morph.size())
				att_score_lis.append(att_score_morph)
				# print(att_score_morph[:,:3])
				# input('...')

			att_score_stack = torch.stack(att_score_lis, dim=1) # b * 63 * 64
			# print(att_score_stack.size())
			# print(att_score_morph[0])
			# print(att_score_stack[0][-1])
			# input('...')

			for idx in range(att_score[0].size(0)): # b

				att_score_arr = np.array(att_score_stack[idx].cpu())
				# print(att_score_arr.shape)
				# input('..')
				att_scores_lis.append(att_score_arr)
				if len(att_scores_lis) == test_set.num_training_sentences:
					print('att matrices recorded: ', len(att_scores_lis))
					break

	att_scores_arr = np.array(att_scores_lis)
	np.save(os.path.join(test_path_out, 'att_score.npy'), att_scores_arr)
	f.write('recorded matrice shape: {}'.format(att_scores_arr.shape))
	f.close()


def print_klloss(test_set, load_dir_af, load_dir_tf, test_path_out, use_gpu, max_seq_len, beam_width, gen_mode='afdynamic'):

	""" 
		print klloss between tf and af models
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir_tf
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model_tf = resume_checkpoint.model
	print('TF Model dir: {}'.format(latest_checkpoint_path))	
	print('TF Model laoded')
	latest_checkpoint_path = load_dir_af
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model_af = resume_checkpoint.model
	print('AF Model dir: {}'.format(latest_checkpoint_path))
	print('AF Model laoded')

	# reset batch_size:
	model_tf.reset_max_seq_len(max_seq_len)
	model_af.reset_max_seq_len(max_seq_len)
	model_tf.reset_use_gpu(use_gpu)	
	model_af.reset_use_gpu(use_gpu)	
	model_tf.reset_batch_size(test_set.batch_size)	
	model_af.reset_batch_size(test_set.batch_size)	
	model_tf.set_beam_width(beam_width)	
	model_af.set_beam_width(beam_width)	
	model_tf.check_classvar('attention_forcing')
	model_af.check_classvar('attention_forcing')
	model_tf.check_classvar('num_unilstm_enc')
	model_af.check_classvar('num_unilstm_enc')
	model_tf.check_classvar('residual')
	model_af.check_classvar('residual')
	model_tf.to(device)
	model_af.to(device)

	print('max seq len {}'.format(model_tf.max_seq_len))
	sys.stdout.flush()

	# load test
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	# f = open(os.path.join(test_path_out, 'test.txt'), 'w')
	model_tf.eval()
	model_af.eval()
	match = 0
	total = 0
	with torch.no_grad():
		for batch in test_batches:

			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)
			non_padding_mask_src = src_ids.data.ne(PAD)
			non_padding_mask_tgt = tgt_ids.data.ne(PAD)

			# import pdb; pdb.set_trace()
			decoder_outputs, decoder_hidden, ret_dict = _forward_aftf(model_tf, model_af, src_ids, tgt=tgt_ids, use_gpu=use_gpu, gen_mode=gen_mode)

			# Get KL loss
			klloss = KLDivLoss()
			attn_hyp = ret_dict['attention_score'] # len-1 * [b x 1 x l]
			attn_ref = ret_dict['attention_ref']
			prev_loss = 0
			losslis = []
			for j in range(len(attn_hyp)):
				# import pdb; pdb.set_trace()
				klloss.eval_batch_with_mask_v2(attn_hyp[j].contiguous(), attn_ref[j].contiguous(), non_padding_mask_tgt[:, j+1])
				loss = klloss.acc_loss.data - prev_loss
				prev_loss = klloss.acc_loss.item()
				losslis.append(loss.item())

			# Evaluation
			seqlist = ret_dict['sequence'] # traverse over time not batch
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			for i in range(len(seqwords)):
				# skip padding sentences in batch (num_sent % batch_size != 0)
				if src_lengths[i] == 0:
					continue
				words = []
				losses = []
				for j in range(len(seqwords[i])):
					word = seqwords[i][j]
					if word == '</s>':
						words.append(word)
						losses.append(round(losslis[j],2))
						break
					if word == '<pad>':
						continue
					else:
						words.append(word)
						losses.append(round(losslis[j],2))
				srcs = _rm_pad(srcwords[i])
				refs = _rm_pad(refwords[i])
				print('src: {}'.format(' '.join(srcs)))
				print('ref: {}'.format(' '.join(refs)))
				print('gen: {}'.format(' '.join(words)))
				print('{}\n'.format(losses))
			sys.stdout.flush()
			input('...')


def record_klloss(test_set, load_dir_af, load_dir_tf, test_path_out, use_gpu, max_seq_len, beam_width, gen_mode='afdynamic'):

	""" 
		print klloss between tf and af models
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir_tf
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model_tf = resume_checkpoint.model
	print('TF Model dir: {}'.format(latest_checkpoint_path))	
	print('TF Model laoded')
	latest_checkpoint_path = load_dir_af
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model_af = resume_checkpoint.model
	print('AF Model dir: {}'.format(latest_checkpoint_path))
	print('AF Model laoded')

	# reset batch_size:
	model_tf.reset_max_seq_len(max_seq_len)
	model_af.reset_max_seq_len(max_seq_len)
	model_tf.reset_use_gpu(use_gpu)	
	model_af.reset_use_gpu(use_gpu)	
	model_tf.reset_batch_size(test_set.batch_size)	
	model_af.reset_batch_size(test_set.batch_size)	
	model_tf.set_beam_width(beam_width)	
	model_af.set_beam_width(beam_width)	
	model_tf.check_classvar('attention_forcing')
	model_af.check_classvar('attention_forcing')
	model_tf.check_classvar('num_unilstm_enc')
	model_af.check_classvar('num_unilstm_enc')
	model_tf.check_classvar('residual')
	model_af.check_classvar('residual')
	model_tf.to(device)
	model_af.to(device)

	print('max seq len {}'.format(model_tf.max_seq_len))
	sys.stdout.flush()

	# load test
	test_batches, vocab_size = test_set.construct_batches(is_train=False)

	# f = open(os.path.join(test_path_out, 'test.txt'), 'w')
	model_tf.eval()
	model_af.eval()
	accum_loss = np.zeros((len(test_batches), model_tf.max_seq_len))
	counter = np.zeros(model_tf.max_seq_len)
	gen_f = open(os.path.join(test_path_out, 'gen.txt'), 'w', encoding="utf8")
	src_f = open(os.path.join(test_path_out, 'src.txt'), 'w', encoding="utf8")
	ref_f = open(os.path.join(test_path_out, 'ref.txt'), 'w', encoding="utf8")
	with torch.no_grad():
		for idx in range(len(test_batches)):
			print('{}/{}'.format(idx,len(test_batches)))
			batch = test_batches[idx]
			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)
			non_padding_mask_src = src_ids.data.ne(PAD)
			non_padding_mask_tgt = tgt_ids.data.ne(PAD)

			# import pdb; pdb.set_trace()
			decoder_outputs, decoder_hidden, ret_dict = _forward_aftf(model_tf, model_af, src_ids, tgt=tgt_ids, use_gpu=use_gpu, gen_mode=gen_mode)

			# Get KL loss
			klloss = KLDivLoss()
			attn_hyp = ret_dict['attention_score'] # len-1 * [b x 1 x l]
			attn_ref = ret_dict['attention_ref']
			prev_loss = 0
			losslis = []
			for j in range(len(attn_hyp)):
				# import pdb; pdb.set_trace()
				klloss.eval_batch_with_mask_v2(attn_hyp[j].contiguous(), attn_ref[j].contiguous(), non_padding_mask_tgt[:, j+1])
				loss = klloss.acc_loss - prev_loss
				prev_loss = klloss.acc_loss.item()
				losslis.append(loss.item())
				accum_loss[idx,j] = round(loss.item(),3)
				if non_padding_mask_tgt[:, j+1] == True:
					counter[j] += 1

			# Evaluation
			seqlist = ret_dict['sequence'] # traverse over time not batch
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			for i in range(len(seqwords)):
				# skip padding sentences in batch (num_sent % batch_size != 0)
				if src_lengths[i] == 0:
					continue
				words = []
				losses = []
				for j in range(len(seqwords[i])):
					word = seqwords[i][j]
					if word == '</s>':
						words.append(word)
						losses.append(round(losslis[j],2))
						break
					elif word == '<pad>':
						continue
					else:
						words.append(word)
						losses.append(round(losslis[j],2))
				srcs = _rm_pad(srcwords[i])
				refs = _rm_pad(refwords[i])
				src_f.write('{}\n'.format(' '.join(srcs)))
				ref_f.write('{}\n'.format(' '.join(refs)))
				gen_f.write('{}\n'.format(' '.join(words)))

	ave_loss = np.round((np.sum(accum_loss, axis=0) / counter) , 5)
	print(ave_loss)
	print(counter)
	np.save(os.path.join(test_path_out, 'att.npy'), accum_loss)
	src_f.close()
	ref_f.close()
	gen_f.close()


def _forward_aftf(model_tf, model_af, src, tgt=None, hidden=None, use_gpu=True, gen_mode='afdynamic'):

	"""
		copied from train.py
		slightly modified the history connection
	"""
	if use_gpu and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
		
	# 00. init
	model_tf.batch_size = model_af.batch_size
	assert model_af.hidden_size_shared == model_tf.hidden_size_shared, \
		'mismatch hidden_size_shared tf:af {}:{}'.format(model_tf.hidden_size_shared,model_af.hidden_size_shared)
	assert model_af.max_seq_len == model_tf.max_seq_len, \
		'mismatch max_seq_len tf:af {}:{}'.format(model_tf.max_seq_len,model_af.max_seq_len)
	batch_size = model_af.batch_size
	hidden_size_shared = model_af.hidden_size_shared
	max_seq_len = model_af.max_seq_len

	# 0. init var for af model
	ret_dict = dict()
	ret_dict[KEY_ATTN_SCORE] = []
	ret_dict[KEY_ATTN_REF] = []
	ret_dict[KEY_SEQUENCE] = []
	ret_dict[KEY_LENGTH] = []
	decoder_outputs = []
	sequence_symbols = []
	sequence_symbols_tf = []

	# 1,2. prep att keys & vals 
	mask_src = src.data.eq(PAD)		
	emb_src_tf, emb_tgt_tf, att_keys_tf, att_vals_tf = model_tf.forward_prep_attkeys(src, tgt, hidden)
	emb_src_af, emb_tgt_af, att_keys_af, att_vals_af = model_af.forward_prep_attkeys(src, tgt, hidden)

	# 3. init hidden states
	dec_hidden = None

	# 4. init for run dec + att + shared + output
	cell_value = torch.FloatTensor([0]).repeat(batch_size, 1, hidden_size_shared).to(device=device)
	prev_c = torch.FloatTensor([0]).repeat(batch_size, 1, max_seq_len).to(device=device) #dummy - for hybrid att only

	tgt_chunk_af = emb_tgt_af[:, 0].unsqueeze(1)
	tgt_chunk_tf = emb_tgt_tf[:, 0].unsqueeze(1)
	cell_value_af = cell_value
	cell_value_tf = cell_value
	prev_c_af = prev_c
	prev_c_tf = prev_c
	dec_hidden_af = dec_hidden
	dec_hidden_tf = dec_hidden
	lengths_tf = np.array([max_seq_len] * batch_size)
	lengths_af = np.array([max_seq_len] * batch_size)


	# 5. for loop over [w1 -> tf -> attref -> af -> w2]
	# note that when using att_tf; w1 does not change att scores: no effect on w2 generation at all
	# bracketd () parts are inactive parts 

	# 5.0 to do tf or not
	for idx in range(max_seq_len - 1):

		# 5.1 gen refatt: [TF] w1 -> tf -> att_tf (-> tf -> w2_tf)
		predicted_softmax_tf, dec_hidden_tf, step_attn_tf, c_out_tf, cell_value_tf = \
			model_tf.forward_step(att_keys_tf, att_vals_tf, tgt_chunk_tf, cell_value_tf, dec_hidden_tf, mask_src, prev_c_tf, use_gpu=use_gpu)
		step_output_tf = predicted_softmax_tf.squeeze(1)
		symbols_tf, lengths_tf, sequence_symbols_tf = model_tf.forward_decode(idx, step_output_tf, lengths_tf, sequence_symbols_tf)
		prev_c_tf = c_out_tf
		# import pdb; pdb.set_trace()

		# 5.2 detach refatt
		step_attn_ref_detach = step_attn_tf.detach() 
		step_attn_ref_detach = step_attn_ref_detach.type(torch.FloatTensor).to(device=device)
		ret_dict[KEY_ATTN_REF].append(step_attn_ref_detach)
		# import pdb; pdb.set_trace()

		# 5.3 gen word prediction: [AF] (w1 -> af -> att_af) att_tf -> af -> w2_af
		predicted_softmax_af, dec_hidden_af, step_attn_af, c_out_af, cell_value_af = \
			model_af.forward_step(att_keys_af, att_vals_af, 
				tgt_chunk_af, cell_value_af, dec_hidden_af, mask_src, prev_c_af, att_ref=step_attn_ref_detach, use_gpu=use_gpu)
		step_output_af = predicted_softmax_af.squeeze(1)
		symbols_af, lengths_af, sequence_symbols = model_tf.forward_decode(idx, step_output_af, lengths_af, sequence_symbols)
		prev_c_af = c_out_af
		# import pdb; pdb.set_trace()

		# 5.4 store var for af model
		ret_dict[KEY_ATTN_SCORE].append(step_attn_af)
		decoder_outputs.append(step_output_af)

		# 5.5 set w2 
		if gen_mode == 'afdynamic':
			tgt_chunk_af = model_af.embedder_dec(symbols_af).to(device=device)
			tgt_chunk_tf = model_tf.embedder_dec(symbols_af).to(device=device)
		elif gen_mode == 'afstatic':
			# tgt_chunk_af = emb_tgt_af[:, idx+1].unsqueeze(1)
			tgt_chunk_af = model_af.embedder_dec(symbols_af).to(device=device)
			tgt_chunk_tf = emb_tgt_tf[:, idx+1].unsqueeze(1).to(device=device)

	# print('...')
	ret_dict[KEY_SEQUENCE] = sequence_symbols
	ret_dict[KEY_LENGTH] = lengths_af.tolist()

	return decoder_outputs, dec_hidden_af, ret_dict	


def _rm_pad(seq):

	outs = []
	for idx in range(len(seq)):
		if seq[idx] == '<pad>':
			continue
		else:
			outs.append(seq[idx])
	return outs



def main():

	# load config
	parser = argparse.ArgumentParser(description='PyTorch Seq2Seq-DD Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	test_path_out = config['test_path_out']
	test_attscore_path = config['test_attscore_path']
	use_teacher = config['use_teacher']
	load_dir = config['load']
	load_dir_af = config['load_af']
	load_dir_tf = config['load_tf']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	gen_mode = config['gen_mode']
	print('attscore dir: {}'.format(test_attscore_path))
	print('output dir: {}'.format(test_path_out))

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)

	config_save_dir = os.path.join(config['test_path_out'], 'eval.cfg') 
	save_config(config, config_save_dir)
	# set test mode: 
	# 3 = DEBUG; 
	# 4 = PLOT -> need to change function params too

	MODE = config['mode']
	if MODE == 3 or MODE == 4:
		max_seq_len = 40
		# max_seq_len = 200
		batch_size = 1
		beam_width = 1
		use_gpu = False

	if MODE == 6: 
		max_seq_len = 64 # same as in training
		# max_seq_len = 200 # or same as in evaluation
		# batch_size = 3
		beam_width = 1

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt,
						path_vocab_src, path_vocab_tgt,
						attscore_path=test_attscore_path,
						max_seq_len=max_seq_len, batch_size=batch_size,
						use_gpu=use_gpu)
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 1:
		# run evaluation
		# print("use gpu: {}".format(config['use_gpu']))
		accuracy = evaluate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width)
		print(accuracy)

	elif MODE == 2: 
		translate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width)

	elif MODE == 3:
		# run debugging
		debug(test_set, load_dir, use_gpu, max_seq_len, beam_width)

	elif MODE == 5:
		# debug for beam search
		debug_beam_search(test_set, load_dir, use_gpu, max_seq_len, beam_width)

	elif MODE == 4:
		# plotting
		att_plot(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, use_teacher)

	elif MODE == 6:
		# write att ref scores
		print_attscore(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, use_teacher)

	elif MODE == 7: 
		translate_att_forcing(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, use_teacher)

	elif MODE == 8: 
		print_klloss(test_set, load_dir_af, load_dir_tf, test_path_out, use_gpu, max_seq_len, beam_width, gen_mode)

	elif MODE == 9: 
		record_klloss(test_set, load_dir_af, load_dir_tf, test_path_out, use_gpu, max_seq_len, beam_width, gen_mode)


if __name__ == '__main__':
	main()

