import torch
# import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

import pickle

sys.path.append('/home/mifs/ytl28/af/af-scripts/')
from utils.misc import set_global_seeds, print_config, save_config, check_srctgt, validate_config, get_memory_alloc
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor, _del_var
from utils.dataset import Dataset
from utils.config import PAD, EOS
from modules.loss import NLLLoss, KLDivLoss, NLLLoss_sched, KLDivLoss_sched, MSELoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import Seq2Seq_DD, Seq2Seq_DD_paf

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger('train')

# logging.basicConfig(filename='app.log', filemode='w')


device = torch.device('cpu')

KEY_ATTN_REF = 'attention_ref'
KEY_ATTN_SCORE = 'attention_score'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'
KEY_MODEL_STRUCT = 'model_struct'


def load_arguments(parser):

	""" lstm based NMT model """
	def none_or_str(value):
		if value == 'None': return None
		else: return value

	# paths
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--train_path_tgt', type=str, required=True, help='train tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--dev_path_tgt', type=str, default=None, help='dev tgt dir')
	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_embedding_src', type=str, default=None, help='pretrained src embedding')
	parser.add_argument('--load_embedding_tgt', type=str, default=None, help='pretrained tgt embedding')
	parser.add_argument('--train_attscore_path', type=str, default=None, help='train set reference attention scores')
	parser.add_argument('--dev_attscore_path', type=str, default=None, help='dev set reference attention scores')

	# model
	parser.add_argument('--embedding_size_enc', type=int, default=200, help='encoder embedding size')
	parser.add_argument('--embedding_size_dec', type=int, default=200, help='decoder embedding size')
	parser.add_argument('--hidden_size_enc', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_bilstm_enc', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--num_unilstm_enc', type=int, default=0, help='number of encoder unilstm layers')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_unilstm_dec', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--hard_att', type=str, default='False', help='use hard attention or not')
	parser.add_argument('--att_mode', type=str, default='bahdanau', \
							help='attention mechanism mode - bahdanau / hybrid / dot_prod')	
	parser.add_argument('--hidden_size_att', type=int, default=1, \
							help='hidden size for bahdanau / hybrid attention')
	parser.add_argument('--hidden_size_shared', type=int, default=200, \
							help='transformed att output hidden size (set as hidden_size_enc)')
	parser.add_argument('--additional_key_size', type=int, default=0, \
							help='additional attention key size: keys = [values, add_feats]')

	# train 
	parser.add_argument('--random_seed', type=int, default=666, help='random seed')	
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')	
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epoches')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--residual', type=str, default='False', help='residual connection')
	parser.add_argument('--max_grad_norm', type=float, default=1.0, help='optimiser gradient norm clipping: max grad norm')	
	parser.add_argument('--batch_first', type=str, default='True', help='batch as the first dimension')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--scheduled_sampling', type=str, default='False', \
					 		help='gradually turn off teacher forcing \
					 		(if True, use teacher_forcing_ratio as the starting point)')

	# teacher forcing / attention forcing / dual
	parser.add_argument('--train_mode', type=str, default='dual', help='train mode; multi | dual | afdynamic')
	parser.add_argument('--load_tf', type=none_or_str, default=None, help='used with train_mode=af; tf model load dir')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='ratio of teacher forcing')
	parser.add_argument('--attention_forcing', type=str, default='False', help='whether or not using attention forcing')
	parser.add_argument('--attention_loss_coeff', type=float, default=1.0, \
							help='attention loss coeff, ignored if attention_forcing=False')
	parser.add_argument('--fr_loss_max_rate', type=float, default=1.0, help='use FR mode if KL_FR < KL_STANDARD rate')
	parser.add_argument('--ep_aaf_start', type=int, default=10, help='AAF starts at this epoch')
	parser.add_argument('--nb_fr_tokens_max', type=int, default=30, help='max nb of free running tokens')

	parser.add_argument('--use_type', type=str, default='word', help='word or char')
	
	# save and print
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')	
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')	

	return parser


class Trainer(object):

	def __init__(self, expt_dir='experiment', 
		load_dir=None,
		loss=NLLLoss(), 
		batch_size=64, 
		random_seed=None,
		checkpoint_every=100, 
		print_every=100, 
		use_gpu=False,
		learning_rate=0.001, 
		max_grad_norm=1.0,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=0.0,
		attention_loss_coeff=1.0,
		attention_forcing=False):

		self.random_seed = random_seed
		if random_seed is not None:
			set_global_seeds(random_seed)

		self.loss = loss
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.use_gpu = use_gpu
		self.learning_rate = learning_rate
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.attention_loss_coeff = attention_loss_coeff
		self.attention_forcing = attention_forcing

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir

		self.batch_size = batch_size
		self.logger = logging.getLogger(__name__)
		# self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)


	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=teacher_forcing_ratio,
												att_key_feats=src_probs, att_scores=attscores)
		# print(len(decoder_outputs))	# max_seq_len - 1

		# Print out intermediate results
		flag_print_inter = False
		if step % self.checkpoint_every == 0 and flag_print_inter:
			seqlist = ret_dict['sequence']
			# convert to words
			srcwords = _convert_to_words_batchfirst(src_ids, model.id2word_enc)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], model.id2word_dec)			
			seqwords = _convert_to_words(seqlist, model.id2word_dec)

			print('---step_res---')
			for i in range(3):
				print('---{}---'.format(i))
				outsrc = 'SRC: {}\n'.format(' '.join(srcwords[i])).encode('utf-8')
				outref = 'REF: {}\n'.format(' '.join(refwords[i])).encode('utf-8')
				outline = 'OUT: {}\n'.format(' '.join(seqwords[i])).encode('utf-8')
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline)
			print('----------------')
			# input('...')
			sys.stdout.flush()

		# Get loss 
		loss.reset()
		for step, step_output in enumerate(decoder_outputs):
			# iterate over seq_len
			if not self.eval_with_mask:
				# print('Train with penalty on mask')
				loss.eval_batch(step_output.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step+1])
				# print(step_output)
				# print(step_output.contiguous().view(self.batch_size, -1))
				# print(tgt_ids[:, step+1])
				# input('...')
			else:
				# print('Train without penalty on mask')
				loss.eval_batch_with_mask(step_output.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step+1], non_padding_mask_tgt[:, step+1])
				# print(step_output) 
				# print(step_output.contiguous().view(self.batch_size, -1).size()) # b * vocab_size
				# print(tgt_ids[:, step+1]) # b * 1
				# print('mask',non_padding_mask_tgt[:, step+1]) # b * 1
				# input('...')
			# print(loss.acc_loss)
			# input('...')

		if self.attention_loss_coeff > 0:

			# Get KL loss
			attn_hyp = ret_dict['attention_score']
			attn_ref = ret_dict['attention_ref']
			# print('sizes ... ')
			# print(len(attn_hyp))
			# print(non_padding_mask_tgt.size())
			# print(attn_hyp[0].size()) # b * 1 * seq_len
			# print(attn_ref[0].size())
			# input('...')
			for idx in range(len(attn_hyp)):
				# print('{} hyp vs ref ...'.format(idx))
				# print(attn_hyp[idx])
				# print(attn_ref[idx])
				# klloss.eval_batch(attn_hyp[idx].contiguous(), attn_ref[idx].contiguous())
				klloss.eval_batch_with_mask_v2(attn_hyp[idx].contiguous(), attn_ref[idx].contiguous(), non_padding_mask_tgt[:, idx+1])
				# print(klloss.acc_loss)
				# input('...')

			# add coeff
			klloss.mul(self.attention_loss_coeff)

			# addition
			total_loss = loss.add(klloss)
			# print(loss.acc_loss, klloss.acc_loss, total_loss)
			# for name, param in model.named_parameters():
			# 	print('grad {}:{}'.format(name, param.grad))
			# 	print('val {}:{}'.format(name, param.data))

			# Backward propagation
			model.zero_grad()
			total_loss.backward()
			# loss.backward(retain_graph=True)
			# klloss.backward()
			resklloss = klloss.get_loss()	

		else:

			# no attention forcing
			model.zero_grad()
			loss.backward()
			resklloss = 0

		self.optimizer.step()
		# input('.........')
		# for name, param in model.named_parameters():
		# 	print('grad {}:{}'.format(name, param.grad))
		# 	print('val {}:{}'.format(name, param.data))

		resloss = loss.get_loss()
		# print(resloss, resklloss)
		# input('...')

		return resloss, resklloss


	def _evaluate_batches(self, model, batches, dataset):

		model.eval()

		loss = self.loss
		loss.reset()
		match = 0
		total = 0

		out_count = 0
		with torch.no_grad():
			for batch in batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)

				decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
														is_training=False,
														att_key_feats=src_probs)

				# Evaluation
				seqlist = other['sequence']
				for step, step_output in enumerate(decoder_outputs):
					target = tgt_ids[:, step+1]
					non_padding = target.ne(PAD)

					if not self.eval_with_mask:
						loss.eval_batch(step_output.view(tgt_ids.size(0), -1), target)
					else:
						loss.eval_batch_with_mask(step_output.view(tgt_ids.size(0), -1), target, non_padding)

					correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()
				
				flag_print_inter=False
				if out_count < 3 and flag_print_inter:
					refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], dataset.tgt_id2word)
					seqwords = _convert_to_words(seqlist, dataset.tgt_id2word)
					outref = 'REF: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
					outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
					sys.stdout.buffer.write(outref)
					sys.stdout.buffer.write(outline)
					out_count += 1


		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		resloss = loss.get_loss()
		torch.cuda.empty_cache()

		return resloss, accuracy
		

	def _train_epoches(self, train_set, model, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch
		print_klloss_total = 0  # Reset every print_every
		epoch_klloss_total = 0  # Reset every epoch

		step = start_step
		step_elapsed = 0
		ckpt = None
		prev_acc = 0.0
		prev_epoch_acc = 0.0

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			# ----------construct batches-----------
			# allow re-shuffling of data
			if type(train_set.attscore_path) != type(None):
				print('--- construct train set (with attscore) ---')
				train_batches, vocab_size = train_set.construct_batches_with_attscore(is_train=True)
			else:
				print('--- construct train set ---')
				train_batches, vocab_size = train_set.construct_batches(is_train=True)

			if dev_set is not None:
				if type(dev_set.attscore_path) != type(None):
					print('--- construct dev set (with attscore) ---')
					dev_batches, vocab_size = dev_set.construct_batches_with_attscore(is_train=False)
				else:
					print('--- construct dev set ---')
					dev_batches, vocab_size = dev_set.construct_batches(is_train=False)


			# --------print info for each epoch----------
			steps_per_epoch = len(train_batches)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))


			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			# self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model.train(True)
			for batch in train_batches:

				# update macro count
				step += 1
				step_elapsed += 1

				# load data
				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				attscores = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				if 'attscores' in batch:
					attscores = batch['attscores'] #list of numpy arrays
					attscores = _convert_to_tensor(attscores, self.use_gpu) #n*31*32

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check src tgt pair ---')
					log_msgs = check_srctgt(src_ids, tgt_ids, train_set.src_id2word, train_set.tgt_id2word)
					for log_msg in log_msgs:
						sys.stdout.buffer.write(log_msg)
						# print(log_msg)

				# convert variable to tensor
				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)
				# print(s rc_probs.size())

				# Get loss
				loss, klloss = self._train_batch(src_ids, tgt_ids, model, step, total_steps, 
												src_probs=src_probs, attscores=attscores)

				print_loss_total += loss
				epoch_loss_total += loss
				print_klloss_total += klloss
				epoch_klloss_total += klloss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					print_klloss_avg = print_klloss_total / self.print_every
					print_klloss_total = 0
					log_msg = 'Progress: %d%%, Train %s: %.4f, att klloss: %.4f,' % (
								step / total_steps * 100,
								self.loss.name,
								print_loss_avg,
								print_klloss_avg)
					# print(log_msg)
					log.info(log_msg)
					# self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)
					# self.writer.add_scalar('train_klloss', print_klloss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps or step == 1:
				# if step == 1:	
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					# save criteria
					if dev_set is not None:
						dev_loss, accuracy = self._evaluate_batches(model, dev_batches, dev_set)
						print('dev loss: {}, accuracy: {}'.format(dev_loss, accuracy))
						model.train(mode=True)

						if prev_acc < accuracy:
						# if True:	
							# save the best model
							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))
							prev_acc = accuracy
							# keep best 5 models
							ckpt.rm_old(self.expt_dir, keep_num=5)

						# else:
						# 	# load last best model - disable [this froze the training..]
						# 	latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
						# 	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
						# 	model = resume_checkpoint.model

					else:
						saved_path = ckpt.save(self.expt_dir)
						print('saving at {} ... '.format(saved_path))
						# keep last 2 models
						ckpt.rm_old(self.expt_dir, keep_num=2)

					# save the last ckpt
					# if step == total_steps:
					# 	saved_path = ckpt.save(self.expt_dir)
					# 	print('saving at {} ... '.format(saved_path))
				
				sys.stdout.flush()

			if step_elapsed == 0: continue
			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			epoch_klloss_avg = epoch_klloss_total / min(steps_per_epoch, step - start_step)
			epoch_klloss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f att klloss: %.4f" % (epoch, self.loss.name, epoch_loss_avg, epoch_klloss_avg)

			# ********************** [finish 1 epoch: eval on dev] ***********************	
			if dev_set is not None:
				# stricter criteria to save if dev set is available - only save when performance improves on dev set
				dev_loss, epoch_accuracy = self._evaluate_batches(model, dev_batches, dev_set)
				self.optimizer.update(dev_loss, epoch)
				# self.writer.add_scalar('dev_loss', dev_loss, global_step=step)
				# self.writer.add_scalar('dev_acc', accuracy, global_step=step)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, epoch_accuracy)
				model.train(mode=True)
				if prev_epoch_acc < epoch_accuracy:
					# save after finishing one epoch
					if ckpt is None:
						ckpt = Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_tgt)

					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					print('saving at {} ... '.format(saved_path))
					prev_epoch_acc = epoch_accuracy			
			else:
				self.optimizer.update(epoch_loss_avg, epoch)
				# save after finishing one epoch
				if ckpt is None:
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)

				saved_path = ckpt.save_epoch(self.expt_dir, epoch)
				print('saving at {} ... '.format(saved_path))			

			log.info('\n')
			log.info(log_msg)


	def train(self, train_set, model, num_epochs=5, resume=False, optimizer=None, dev_set=None):

		""" 
			Run training for a given model.
			Args:
				train_set: Dataset
				dev_set: Dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run (default 5)
				resume(bool, optional): resume training with the latest checkpoint, (default False)
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
				   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
				
			Returns:
				model (seq2seq.models): trained model.
		"""

		torch.cuda.empty_cache()
		self.resume = resume
		if resume:
			latest_checkpoint_path = Checkpoint.get_latest_epoch_checkpoint(self.load_dir)
			print('resuming {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			self.optimizer = resume_checkpoint.optimizer

			# check var
			model.set_var('attention_forcing', self.attention_forcing)
			model.set_var('debug_count', 0)
			model.reset_use_gpu(self.use_gpu)
			print('attention forcing: {}'.format(model.attention_forcing))
			print('use gpu: {}'.format(model.use_gpu))
			if self.use_gpu:
				model = model.cuda()
			else:
				model = model.cpu()

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

		else:
			start_epoch = 1
			step = 0

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))
				# check embedder init
				# if 'embedder' in name:
				# 	print('{}:{}'.format(name, param[5]))

			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(model.parameters(), 
							lr=self.learning_rate), max_grad_norm=self.max_grad_norm) # 5 -> 1

				# set scheduler
				# optimizer.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'min'))

			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(train_set, model, num_epochs, start_epoch, step, dev_set=dev_set)
		
		return model



class Trainer_aaf_base(Trainer):
	"""same as Trainer, except for rmoving 'for loop' used to compute loss"""
	def __init__(self, load_tf_dir=None, **kwargs):
		super(Trainer_aaf_base, self).__init__(**kwargs)
		self.load_tf_dir = load_tf_dir

		# for aaf
		# self.dct_fr_seqs = {}
		self.dct_info = {'nll':[], 'kl':[], 'fr_percent':[]}
		
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""
		# import pdb; pdb.set_trace()

		# define loss
		loss = self.loss
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			ss_max = 0.2 # 0.2 0.1

			# linear schedule
			teacher_forcing_ratio = 1.0 - ss_max * progress

			# Inverse sigmoid decay
			# ss_k = 10 # 5
			# teacher_forcing_ratio = 1.0 -ss_max + ss_max * (ss_k / (ss_k + np.exp(progress * 100 / ss_k)))

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=teacher_forcing_ratio,
												att_key_feats=src_probs, att_scores=attscores)
		# print('aaf_base')
		# # print(len(decoder_outputs))	# max_seq_len - 1

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		loss.reset()

		if not self.eval_with_mask:
			loss.eval_batch_seq(torch.stack(decoder_outputs, dim=2), 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(torch.stack(decoder_outputs, dim=2), 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])

		if np.isnan(loss.get_loss()):
			print('loss', loss.get_loss())
			print('klloss', klloss.get_loss())
			print('src', src_ids)
			print('tgt', tgt_ids)
			print('out[0]', decoder_outputs[0])
			for name, param in model.named_parameters():
				print('grad {}:{}'.format(name, param.grad))
				print('val {}:{}'.format(name, param.data))
			import pdb; pdb.set_trace()


		# for step, step_output in enumerate(decoder_outputs):
		# 	# iterate over seq_len
		# 	if not self.eval_with_mask:
		# 		# print('Train with penalty on mask')
		# 		loss.eval_batch(step_output.contiguous()\
		# 			.view(self.batch_size, -1), tgt_ids[:, step+1])
		# 	else:
		# 		# print('Train without penalty on mask')
		# 		loss.eval_batch_with_mask(step_output.contiguous()\
		# 			.view(self.batch_size, -1), tgt_ids[:, step+1], non_padding_mask_tgt[:, step+1])
		# print(loss.get_loss())


		if self.attention_loss_coeff > 0:

			# Get KL loss
			attn_hyp = ret_dict['attention_score']
			attn_ref = ret_dict['attention_ref']

			klloss.eval_batch_seq_with_mask(torch.cat(attn_hyp,dim=1), 
									torch.cat(attn_ref,dim=1), 
									non_padding_mask_tgt[:, 1:])

			# for idx in range(len(attn_hyp)):
			# 	# klloss.eval_batch(attn_hyp[idx].contiguous(), attn_ref[idx].contiguous())
			# 	klloss.eval_batch_with_mask_v2(attn_hyp[idx].contiguous(), attn_ref[idx].contiguous(), non_padding_mask_tgt[:, idx+1])
			# print(klloss.get_loss())

			# add coeff
			klloss.mul(self.attention_loss_coeff)

			# addition
			total_loss = loss.add(klloss)
			# print(loss.acc_loss, klloss.acc_loss, total_loss)
			# for name, param in model.named_parameters():
			# 	print('grad {}:{}'.format(name, param.grad))
			# 	print('val {}:{}'.format(name, param.data))

			# Backward propagation
			model.zero_grad()
			total_loss.backward()
			# loss.backward(retain_graph=True)
			# klloss.backward()
			resklloss = klloss.get_loss()	

		else:

			# no attention forcing
			model.zero_grad()
			loss.backward()
			resklloss = 0

		self.optimizer.step()
		# input('.........')
		# cnt = 0
		# for name, param in model.named_parameters():
		# 	print('grad {}:{}'.format(name, param.grad))
		# 	print('val {}:{}'.format(name, param.data))
		# 	cnt +=1
		# 	if cnt>3: break

		resloss = loss.get_loss()
		# print(resloss, resklloss, klloss.get_fr_percent())
		# import pdb; pdb.set_trace()
		# input('...')
		fr_percent = 1.0 - teacher_forcing_ratio
		self._update_dct_info(step, resloss, resklloss, fr_percent)

		return resloss, resklloss

	def _update_dct_info(self, step, resloss, resklloss, fr_percent=-1.0):
		self.dct_info['nll'].append(resloss)
		self.dct_info['kl'].append(resklloss)
		self.dct_info['fr_percent'].append(fr_percent)
		flag_print_inter = True
		if step % self.checkpoint_every == 0 and flag_print_inter:
			dirFile = os.path.join(self.expt_dir,'dct_info.pkl')
			with open(dirFile,'wb') as f:
				pickle.dump(self.dct_info, f, protocol=2)

	def _print_seqs(self, model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=None):
		def _pure_txt(lst, lst_rm=['<s>', '</s>', '<pad>']):
			return [l for l in lst if l not in lst_rm]

		# Print out intermediate results
		flag_print_inter = True
		if step % self.checkpoint_every == 0 and flag_print_inter:
			# convert to words
			srcwords = _convert_to_words_batchfirst(src_ids, model.id2word_enc)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], model.id2word_dec)			
			seqwords = _convert_to_words(ret_dict['sequence'], model.id2word_dec)
			if ret_dict_fr is not None: seqwords_fr = _convert_to_words(ret_dict_fr['sequence'], model.id2word_dec)

			print('---step_res---')
			for i in range(1):
				outsrc = 'SRC: {}\n'.format(' '.join(_pure_txt(srcwords[i]))).encode('utf-8')
				outref = 'REF: {}\n'.format(' '.join(_pure_txt(refwords[i]))).encode('utf-8')
				outline = 'OUT: {}\n'.format(' '.join(_pure_txt(seqwords[i]))).encode('utf-8')
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline)
				if ret_dict_fr is not None:
					sys.stdout.buffer.write('OUT_FR: {}\n'.format(' '.join(_pure_txt(seqwords_fr[i]))).encode('utf-8'))
			print('----------------')
			# input('...')
			sys.stdout.flush()

	def train(self, train_set, model, **kwargs):
		if self.load_tf_dir is not None:
			dropout_rate = model.dropout_rate
			print('dropout {}'.format(dropout_rate))

			# load tf model
			latest_checkpoint_path = self.load_tf_dir
			resume_checkpoint_tf = Checkpoint.load(latest_checkpoint_path)
			# model = resume_checkpoint_tf.model.to(device)
			model.load_state_dict(resume_checkpoint_tf.model.state_dict())
			model.to(device)
			
			model.reset_dropout(dropout_rate)
			model.reset_use_gpu(self.use_gpu)
			model.reset_batch_size(self.batch_size)

			print('TF / pretrained Model dir: {}'.format(latest_checkpoint_path))
			print('TF / pretrained Model loaded')

		super(Trainer_aaf_base, self).train(train_set, model, **kwargs)

		

class Trainer_aaf(Trainer_aaf_base):
	def __init__(self, load_tf_dir=None, fr_loss_max_rate=1.0, ep_aaf_start=10, **kwargs):
		super(Trainer_aaf_base, self).__init__(**kwargs)
		self.load_tf_dir = load_tf_dir

		# for aaf
		self.fr_loss_max_rate = fr_loss_max_rate
		self.ep_aaf_start = ep_aaf_start
		self.dct_fr_seqs = {}
		self.dct_info = {'nll':[], 'kl':[], 'fr_percent':[]}

	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None, epoch=None):
		if epoch < self.ep_aaf_start:
			return super(Trainer_aaf, self)._train_batch(src_ids, tgt_ids, model, step, total_steps, src_probs=src_probs, attscores=attscores)
		else:
			return self._train_batch_aaf(src_ids, tgt_ids, model, step, total_steps, src_probs=src_probs, attscores=attscores, epoch=epoch)

	def _train_batch_aaf(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None, epoch=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss(fr_loss_max_rate=self.fr_loss_max_rate, ep_aaf_start=self.ep_aaf_start)

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		# get 2 versions: tf model always uses ref history, af model uses ref / gen history
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attscores)

		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=0.0,
												att_key_feats=src_probs, att_scores=attscores)

		# import pdb; pdb.set_trace()
		# decoder_outputs = torch.stack(decoder_outputs, dim=2)
		# print(tgt_ids.size(), tgt_ids[0])
		# print(decoder_outputs.size(), decoder_outputs[0])

		# print('aaf')
		# # print(len(decoder_outputs))	# max_seq_len - 1

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# again, 2 versions; att loss first, then output and total

		# compare KL, choose which loss to use using masks
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		# Get KL loss
		attn_hyp = ret_dict['attention_score']
		attn_ref = ret_dict['attention_ref']

		attn_hyp_fr = ret_dict_fr['attention_score']
		attn_ref_fr = ret_dict_fr['attention_ref']

		# import pdb; pdb.set_trace()
		# attn_ref = torch.cat(attn_ref,dim=1)
		# attn_hyp = torch.cat(attn_hyp,dim=1)
		# print(attn_ref.size(), attn_ref[0])
		# print(attn_hyp.size(), attn_hyp[0])
		# pdb.set_trace()
		# print(non_padding_mask_tgt.size())

		# TODO: rm the (non-)stack option
		if model.flag_stack_outputs:
			klloss.eval_batch_seq_with_mask_smooth(attn_hyp, 
										attn_ref, 
										non_padding_mask_tgt[:, 1:],
										attn_hyp_fr,
										epoch)
		else:
			klloss.eval_batch_seq_with_mask_smooth(torch.cat(attn_hyp,dim=1), 
										torch.cat(attn_ref,dim=1), 
										non_padding_mask_tgt[:, 1:],
										torch.cat(attn_hyp_fr,dim=1),
										epoch)
		

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# import pdb; pdb.set_trace()

		loss.reset()
		if model.flag_stack_outputs:
			if not self.eval_with_mask:
				loss.eval_batch_seq(decoder_outputs, 
					tgt_ids[:, 1:], 
					decoder_outputs_fr,
					klloss.mask_utt_fr)
			else:
				loss.eval_batch_seq_with_mask(decoder_outputs, 
					tgt_ids[:, 1:], 
					non_padding_mask_tgt[:, 1:],
					decoder_outputs_fr,
					klloss.mask_utt_fr)
		else:
			if not self.eval_with_mask:
				loss.eval_batch_seq(torch.stack(decoder_outputs, dim=2), 
					tgt_ids[:, 1:], 
					torch.stack(decoder_outputs_fr, dim=2),
					klloss.mask_utt_fr)
			else:
				loss.eval_batch_seq_with_mask(torch.stack(decoder_outputs, dim=2), 
					tgt_ids[:, 1:], 
					non_padding_mask_tgt[:, 1:],
					torch.stack(decoder_outputs_fr, dim=2),
					klloss.mask_utt_fr)
		


		# addition
		total_loss = loss.add(klloss)

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		# loss.backward(retain_graph=True)
		# klloss.backward()
		resklloss = klloss.get_loss()


		self.optimizer.step()

		resloss = loss.get_loss()
		fr_percent = klloss.get_fr_percent()
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resklloss, fr_percent)
		self._update_dct_fr_seqs(model, step, src_ids, tgt_ids, ret_dict, klloss.mask_utt_fr, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss, fr_percent


	def _update_dct_fr_seqs(self, model, step, src_ids, tgt_ids, ret_dict, switch_utt_fr, ret_dict_fr=None):
		# import pdb; pdb.set_trace()
		# get src_seq_lst
		# get switch_utt_fr
		# update dct_fr_seqs
		def _pure_txt(lst, lst_rm=['<s>', '</s>', '<pad>']):
			return [l for l in lst if l not in lst_rm]
		src_seq_lst = _convert_to_words_batchfirst(src_ids, model.id2word_enc)
		src_seq_lst = [' '.join(_pure_txt(s)).encode('utf-8') for i,s in enumerate(src_seq_lst) if switch_utt_fr[i]==1]
		for s in src_seq_lst:
			if not (s in self.dct_fr_seqs.keys()):
				self.dct_fr_seqs[s] = 1
			else:
				self.dct_fr_seqs[s] += 1

		# Print out intermediate results
		flag_print_inter = True
		if step % self.checkpoint_every == 0 and flag_print_inter:
			# convert to words
			srcwords = _convert_to_words_batchfirst(src_ids, model.id2word_enc)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], model.id2word_dec)			
			seqwords = _convert_to_words(ret_dict['sequence'], model.id2word_dec)
			if ret_dict_fr is not None: seqwords_fr = _convert_to_words(ret_dict_fr['sequence'], model.id2word_dec)

			print('---step_res---')
			for i in range(1):
				outsrc = 'SRC: {}\n'.format(' '.join(_pure_txt(srcwords[i]))).encode('utf-8')
				outref = 'REF: {}\n'.format(' '.join(_pure_txt(refwords[i]))).encode('utf-8')
				outline = 'OUT: {}\n'.format(' '.join(_pure_txt(seqwords[i]))).encode('utf-8')
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline)
				if ret_dict_fr is not None:
					sys.stdout.buffer.write('OUT_FR: {}\n'.format(' '.join(_pure_txt(seqwords_fr[i]))).encode('utf-8'))
			print('----------------')
			# input('...')
			sys.stdout.flush()

			# save pkl
			dirFile = os.path.join(self.expt_dir,'dct_fr_seqs.pkl')
			with open(dirFile,'wb') as f:
				pickle.dump(self.dct_fr_seqs, f, protocol=2)

	def _train_epoches(self, train_set, model, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch
		print_klloss_total = 0  # Reset every print_every
		epoch_klloss_total = 0  # Reset every epoch
		print_fr_percent_total = 0  # Reset every print_every
		epoch_fr_percent_total = 0  # Reset every epoch

		step = start_step
		step_elapsed = 0
		ckpt = None
		prev_acc = 0.0
		prev_epoch_acc = 0.0

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			# ----------construct batches-----------
			# allow re-shuffling of data
			if type(train_set.attscore_path) != type(None):
				print('--- construct train set (with attscore) ---')
				train_batches, vocab_size = train_set.construct_batches_with_attscore(is_train=True)
			else:
				print('--- construct train set ---')
				train_batches, vocab_size = train_set.construct_batches(is_train=True)

			if dev_set is not None:
				if type(dev_set.attscore_path) != type(None):
					print('--- construct dev set (with attscore) ---')
					dev_batches, vocab_size = dev_set.construct_batches_with_attscore(is_train=False)
				else:
					print('--- construct dev set ---')
					dev_batches, vocab_size = dev_set.construct_batches(is_train=False)


			# --------print info for each epoch----------
			steps_per_epoch = len(train_batches)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))


			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			# self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model.train(True)

			for batch in train_batches:

				# update macro count
				step += 1
				step_elapsed += 1

				# load data
				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				attscores = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				if 'attscores' in batch:
					attscores = batch['attscores'] #list of numpy arrays
					attscores = _convert_to_tensor(attscores, self.use_gpu) #n*31*32

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check src tgt pair ---')
					log_msgs = check_srctgt(src_ids, tgt_ids, train_set.src_id2word, train_set.tgt_id2word)
					for log_msg in log_msgs:
						sys.stdout.buffer.write(log_msg)
						# print(log_msg)

				# convert variable to tensor
				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)
				# print(src_probs.size())

				# Get loss
				# import pdb; pdb.set_trace()
				tmp = self._train_batch(src_ids, tgt_ids, model, step, total_steps, 
												src_probs=src_probs, attscores=attscores, epoch=epoch)
				if len(tmp)==2:
					loss, klloss = tmp
					fr_percent = -1.0
				else:
					loss, klloss, fr_percent = tmp

				print_loss_total += loss
				epoch_loss_total += loss
				print_klloss_total += klloss
				epoch_klloss_total += klloss
				print_fr_percent_total += fr_percent
				epoch_fr_percent_total += fr_percent

				if step % self.print_every == 0 and step_elapsed >= self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					print_klloss_avg = print_klloss_total / self.print_every
					print_klloss_total = 0
					print_fr_percent_avg = print_fr_percent_total / self.print_every
					print_fr_percent_total = 0
					log_msg = 'Progress: %d%%, Train %s: %.4f, att klloss: %.4f,' % (
								step / total_steps * 100,
								self.loss.name,
								print_loss_avg,
								print_klloss_avg)
					# print(log_msg)
					log.info(log_msg)
					log.info('TMP fr_percent: {}'.format(print_fr_percent_avg))
					
					# self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)
					# self.writer.add_scalar('train_klloss', print_klloss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps or step == 1:
				# if step == 1:	
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					# save criteria
					if dev_set is not None:
						dev_loss, accuracy = self._evaluate_batches(model, dev_batches, dev_set)
						print('dev loss: {}, accuracy: {}'.format(dev_loss, accuracy))
						model.train(mode=True)

						if prev_acc < accuracy:
						# if True:	
							# save the best model
							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))
							prev_acc = accuracy
							# keep best 5 models
							ckpt.rm_old(self.expt_dir, keep_num=5)

						# else:
						# 	# load last best model - disable [this froze the training..]
						# 	latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
						# 	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
						# 	model = resume_checkpoint.model

					else:
						saved_path = ckpt.save(self.expt_dir)
						print('saving at {} ... '.format(saved_path))
						# keep last 2 models
						ckpt.rm_old(self.expt_dir, keep_num=2)

					# save the last ckpt
					# if step == total_steps:
					# 	saved_path = ckpt.save(self.expt_dir)
					# 	print('saving at {} ... '.format(saved_path))
				
				sys.stdout.flush()


			if step_elapsed == 0: continue
			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			epoch_klloss_avg = epoch_klloss_total / min(steps_per_epoch, step - start_step)
			epoch_klloss_total = 0
			epoch_fr_percent_avg = epoch_fr_percent_total / min(steps_per_epoch, step - start_step)
			epoch_fr_percent_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f att klloss: %.4f; free run percent %.4f" % (epoch, self.loss.name, epoch_loss_avg, epoch_klloss_avg, epoch_fr_percent_avg)

			# ********************** [finish 1 epoch: eval on dev] ***********************	
			if dev_set is not None:
				# stricter criteria to save if dev set is available - only save when performance improves on dev set
				dev_loss, epoch_accuracy = self._evaluate_batches(model, dev_batches, dev_set)
				self.optimizer.update(dev_loss, epoch)
				# self.writer.add_scalar('dev_loss', dev_loss, global_step=step)
				# self.writer.add_scalar('dev_acc', accuracy, global_step=step)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, epoch_accuracy)
				model.train(mode=True)
				if prev_epoch_acc < epoch_accuracy:
					# save after finishing one epoch
					if ckpt is None:
						ckpt = Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_tgt)

					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					print('saving at {} ... '.format(saved_path))
					prev_epoch_acc = epoch_accuracy			
			else:
				self.optimizer.update(epoch_loss_avg, epoch)
				# save after finishing one epoch
				if ckpt is None:
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)

				saved_path = ckpt.save_epoch(self.expt_dir, epoch)
				print('saving at {} ... '.format(saved_path))			

			log.info('\n')
			log.info(log_msg)


class Trainer_aoaf(Trainer_aaf):
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None, epoch=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""
		# import pdb; pdb.set_trace()

		# define loss
		loss = self.loss
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		# get 2 versions: tf model always uses ref history, af model uses tf history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# again, 2 versions; att loss first, then output and total

		# compare KL, choose which loss to use using masks
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		# Get KL loss
		klloss.eval_batch_seq_with_mask_smooth_aoaf(attn_hyp_fr, 
									attn_hyp, 
									non_padding_mask_tgt[:, 1:],
									teacher_forcing_ratio)
		
		# add coeff
		klloss.mul(self.attention_loss_coeff)

		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:], 
				decoder_outputs_fr,
				klloss.mask_utt_fr)
		else:
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:],
				decoder_outputs_fr,
				klloss.mask_utt_fr)

		# addition
		fr_percent = 1.0 - teacher_forcing_ratio
		total_loss = loss.acc_loss + klloss.acc_loss

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resklloss = klloss.get_loss()
		
		self._update_dct_info(step, resloss, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss, fr_percent


class Trainer_paf(Trainer_aaf_base):
	def __init__(self, nb_fr_tokens_max=30, **kwargs):
		super(Trainer_paf, self).__init__(**kwargs)
		self.nb_fr_tokens_max = nb_fr_tokens_max

	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None, epoch=None, tgt_lengths=None):
		
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""
		# import pdb; pdb.set_trace()

		# define loss
		loss = self.loss
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# partial free running
		# if self.partial_free_running:
		nb_fr_tokens = int(self.nb_fr_tokens_max * step / total_steps)+1
		nb_tf_tokens = max(tgt_lengths) - nb_fr_tokens
		# import pdb; pdb.set_trace()

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=teacher_forcing_ratio,
												att_key_feats=src_probs, att_scores=attscores, 
												nb_tf_tokens=nb_tf_tokens)

		# Get loss 
		loss.reset()

		if not self.eval_with_mask:
			loss.eval_batch_seq(torch.stack(decoder_outputs, dim=2), 
				tgt_ids[:, 1:])
		else:
			loss.eval_batch_seq_with_mask(torch.stack(decoder_outputs, dim=2), 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])

		if np.isnan(loss.get_loss()):
			print('loss', loss.get_loss())
			print('klloss', klloss.get_loss())
			print('src', src_ids)
			print('tgt', tgt_ids)
			print('out[0]', decoder_outputs[0])
			for name, param in model.named_parameters():
				print('grad {}:{}'.format(name, param.grad))
				print('val {}:{}'.format(name, param.data))
			import pdb; pdb.set_trace()



		if self.attention_loss_coeff > 0:
			# Get KL loss
			attn_hyp = ret_dict['attention_score']
			attn_ref = ret_dict['attention_ref']

			klloss.eval_batch_seq_with_mask(torch.cat(attn_hyp,dim=1), 
									torch.cat(attn_ref,dim=1), 
									non_padding_mask_tgt[:, 1:])

			# add coeff
			klloss.mul(self.attention_loss_coeff)

			# addition
			total_loss = loss.add(klloss)
			# print(loss.acc_loss, klloss.acc_loss, total_loss)

			# Backward propagation
			model.zero_grad()
			total_loss.backward()
			# loss.backward(retain_graph=True)
			# klloss.backward()
			resklloss = klloss.get_loss()	

		else:
			# no attention forcing
			model.zero_grad()
			loss.backward()
			resklloss = 0

		self.optimizer.step()

		resloss = loss.get_loss()
		# print(resloss, resklloss, klloss.get_fr_percent())
		# import pdb; pdb.set_trace()
		self._update_dct_info(step, resloss, resklloss, nb_fr_tokens)

		return resloss, resklloss, nb_fr_tokens

	def _train_epoches(self, train_set, model, n_epochs, start_epoch, start_step, dev_set=None):
		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch
		print_klloss_total = 0  # Reset every print_every
		epoch_klloss_total = 0  # Reset every epoch
		print_fr_percent_total = 0  # Reset every print_every
		epoch_fr_percent_total = 0  # Reset every epoch

		step = start_step
		step_elapsed = 0
		ckpt = None
		prev_acc = 0.0
		prev_epoch_acc = 0.0

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			# ----------construct batches-----------
			# allow re-shuffling of data
			if type(train_set.attscore_path) != type(None):
				print('--- construct train set (with attscore) ---')
				train_batches, vocab_size = train_set.construct_batches_with_attscore_sort(is_train=True)
			else:
				print('--- construct train set ---')
				train_batches, vocab_size = train_set.construct_batches(is_train=True)

			if dev_set is not None:
				if type(dev_set.attscore_path) != type(None):
					print('--- construct dev set (with attscore) ---')
					dev_batches, vocab_size = dev_set.construct_batches_with_attscore(is_train=False)
				else:
					print('--- construct dev set ---')
					dev_batches, vocab_size = dev_set.construct_batches(is_train=False)


			# --------print info for each epoch----------
			steps_per_epoch = len(train_batches)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))


			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			# self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model.train(True)

			# cnt = 0
			# for batch in train_batches:
			# 	cnt+=1
			# 	src_ids = batch['src_word_ids']
			# 	src_lengths = batch['src_sentence_lengths']
			# 	tgt_ids = batch['tgt_word_ids']
			# 	tgt_lengths = batch['tgt_sentence_lengths']
			# 	if cnt%1000==0:
			# 		print(src_lengths)
			# 		print(tgt_lengths)
			# 		print(src_ids[0])
			# 		print(tgt_ids[0])
			# 		pdb.set_trace()


			for batch in train_batches:

				# update macro count
				step += 1
				step_elapsed += 1

				# load data
				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				attscores = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				if 'attscores' in batch:
					attscores = batch['attscores'] #list of numpy arrays
					attscores = _convert_to_tensor(attscores, self.use_gpu) #n*31*32


				# pdb.set_trace()

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check src tgt pair ---')
					log_msgs = check_srctgt(src_ids, tgt_ids, train_set.src_id2word, train_set.tgt_id2word)
					for log_msg in log_msgs:
						sys.stdout.buffer.write(log_msg)
						# print(log_msg)

				# convert variable to tensor
				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)
				# print(src_probs.size())

				# Get loss
				# import pdb; pdb.set_trace()
				tmp = self._train_batch(src_ids, tgt_ids, model, step, total_steps, 
												src_probs=src_probs, attscores=attscores, epoch=epoch, tgt_lengths=tgt_lengths)
				if len(tmp)==2:
					loss, klloss = tmp
					fr_percent = -1.0
				else:
					loss, klloss, fr_percent = tmp

				print_loss_total += loss
				epoch_loss_total += loss
				print_klloss_total += klloss
				epoch_klloss_total += klloss
				print_fr_percent_total += fr_percent
				epoch_fr_percent_total += fr_percent

				if step % self.print_every == 0 and step_elapsed >= self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					print_klloss_avg = print_klloss_total / self.print_every
					print_klloss_total = 0
					print_fr_percent_avg = print_fr_percent_total / self.print_every
					print_fr_percent_total = 0
					log_msg = 'Progress: %d%%, Train %s: %.4f, att klloss: %.4f,' % (
								step / total_steps * 100,
								self.loss.name,
								print_loss_avg,
								print_klloss_avg)
					# print(log_msg)
					log.info(log_msg)
					log.info('TMP nb_fr_tokens: {}'.format(print_fr_percent_avg))
					
					# self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)
					# self.writer.add_scalar('train_klloss', print_klloss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps or step == 1:
				# if step == 1:	
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					# save criteria
					if dev_set is not None:
						dev_loss, accuracy = self._evaluate_batches(model, dev_batches, dev_set)
						print('dev loss: {}, accuracy: {}'.format(dev_loss, accuracy))
						model.train(mode=True)

						if prev_acc < accuracy:
						# if True:	
							# save the best model
							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))
							prev_acc = accuracy
							# keep best 5 models
							ckpt.rm_old(self.expt_dir, keep_num=5)

						# else:
						# 	# load last best model - disable [this froze the training..]
						# 	latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
						# 	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
						# 	model = resume_checkpoint.model

					else:
						saved_path = ckpt.save(self.expt_dir)
						print('saving at {} ... '.format(saved_path))
						# keep last 2 models
						ckpt.rm_old(self.expt_dir, keep_num=2)

					# save the last ckpt
					# if step == total_steps:
					# 	saved_path = ckpt.save(self.expt_dir)
					# 	print('saving at {} ... '.format(saved_path))
				
				sys.stdout.flush()


			if step_elapsed == 0: continue
			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			epoch_klloss_avg = epoch_klloss_total / min(steps_per_epoch, step - start_step)
			epoch_klloss_total = 0
			epoch_fr_percent_avg = epoch_fr_percent_total / min(steps_per_epoch, step - start_step)
			epoch_fr_percent_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f att klloss: %.4f; free run tokens %.4f" % (epoch, self.loss.name, epoch_loss_avg, epoch_klloss_avg, epoch_fr_percent_avg)

			# ********************** [finish 1 epoch: eval on dev] ***********************	
			if dev_set is not None:
				# stricter criteria to save if dev set is available - only save when performance improves on dev set
				dev_loss, epoch_accuracy = self._evaluate_batches(model, dev_batches, dev_set)
				self.optimizer.update(dev_loss, epoch)
				# self.writer.add_scalar('dev_loss', dev_loss, global_step=step)
				# self.writer.add_scalar('dev_acc', accuracy, global_step=step)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, epoch_accuracy)
				model.train(mode=True)
				if prev_epoch_acc < epoch_accuracy:
					# save after finishing one epoch
					if ckpt is None:
						ckpt = Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_tgt)

					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					print('saving at {} ... '.format(saved_path))
					prev_epoch_acc = epoch_accuracy			
			else:
				self.optimizer.update(epoch_loss_avg, epoch)
				# save after finishing one epoch
				if ckpt is None:
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)

				saved_path = ckpt.save_epoch(self.expt_dir, epoch)
				print('saving at {} ... '.format(saved_path))			

			log.info('\n')
			log.info(log_msg)


class Trainer_oaf_v0_nosched(Trainer_aaf_base):
	def __init__(self, **kwargs):
		super(Trainer_oaf, self).__init__(**kwargs)
		self.dct_info = {'nll':[], 'nll_fr':[], 'kl':[], 'fr_percent':[]}

	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()
		nll_fr = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		# import pdb; pdb.set_trace()
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		# print(decoder_samples.size(), decoder_samples[:5,:10])
		# print(tgt_ids.size(), tgt_ids[:5,:10])
		# pdb.set_trace()
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:])
			nll_fr.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
			nll_fr.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		
		# Get KL loss
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		klloss.eval_batch_seq_with_mask(attn_hyp_fr, 
										attn_hyp.detach(), 
										non_padding_mask_tgt[:, 1:]
										)

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# addition
		# total_loss = loss.add(klloss)
		total_loss = loss.acc_loss + (nll_fr.acc_loss + klloss.acc_loss)

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resloss_fr = nll_fr.get_loss()
		resklloss = klloss.get_loss()
		fr_percent = 1.0 - teacher_forcing_ratio
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resloss_fr, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss

	def _update_dct_info(self, step, resloss, resloss_fr, resklloss, fr_percent=-1.0):
		self.dct_info['nll'].append(resloss)
		self.dct_info['nll_fr'].append(resloss_fr)
		self.dct_info['kl'].append(resklloss)
		self.dct_info['fr_percent'].append(fr_percent)
		flag_print_inter = True
		if step % self.checkpoint_every == 0 and flag_print_inter:
			dirFile = os.path.join(self.expt_dir,'dct_info.pkl')
			with open(dirFile,'wb') as f:
				pickle.dump(self.dct_info, f, protocol=2)

class Trainer_oaf_v1_notf(Trainer_aaf_base):
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		with torch.no_grad():
			decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
													is_training=True, 
													teacher_forcing_ratio=1.0,
													att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			# loss.eval_batch_seq(decoder_outputs, 
			# 	tgt_ids[:, 1:])
			loss.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# loss.eval_batch_seq_with_mask(decoder_outputs, 
			# 	tgt_ids[:, 1:], 
			# 	non_padding_mask_tgt[:, 1:])
			loss.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		# loss.acc_loss *= 0.5
		
		# Get KL loss
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		klloss.eval_batch_seq_with_mask(attn_hyp_fr, 
										attn_hyp.detach(), 
										non_padding_mask_tgt[:, 1:]
										)

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# addition
		total_loss = loss.add(klloss)

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resklloss = klloss.get_loss()
		fr_percent = 1.0 - teacher_forcing_ratio
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resklloss, fr_percent)

		return resloss, resklloss

class Trainer_oaf(Trainer_aaf_base):
	def __init__(self, **kwargs):
		super(Trainer_oaf, self).__init__(**kwargs)
		self.dct_info = {'nll':[], 'nll_fr':[], 'kl':[], 'fr_percent':[]}

	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()
		nll_fr = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# fixed
			teacher_forcing_ratio = 0.5
			# linear schedule
			# teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		# import pdb; pdb.set_trace()
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		# print(decoder_samples.size(), decoder_samples[:5,:10])
		# print(tgt_ids.size(), tgt_ids[:5,:10])
		# pdb.set_trace()
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:])
			nll_fr.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
			nll_fr.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		# nll_fr.mul(1.0 - teacher_forcing_ratio)
		
		# Get KL loss
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		klloss.eval_batch_seq_with_mask(attn_hyp_fr, 
										attn_hyp.detach(), 
										non_padding_mask_tgt[:, 1:]
										)

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# addition
		# total_loss = loss.add(klloss)

		# oaf loss
		fr_percent = 1.0 - teacher_forcing_ratio
		total_loss = (1.0 - fr_percent) * loss.acc_loss + fr_percent * (nll_fr.acc_loss + klloss.acc_loss)

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resloss_fr = nll_fr.get_loss()
		resklloss = klloss.get_loss()
		
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resloss_fr, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss

	def _update_dct_info(self, step, resloss, resloss_fr, resklloss, fr_percent=-1.0):
		self.dct_info['nll'].append(resloss)
		self.dct_info['nll_fr'].append(resloss_fr)
		self.dct_info['kl'].append(resklloss)
		self.dct_info['fr_percent'].append(fr_percent)
		flag_print_inter = True
		if step % self.checkpoint_every == 0 and flag_print_inter:
			dirFile = os.path.join(self.expt_dir,'dct_info.pkl')
			with open(dirFile,'wb') as f:
				pickle.dump(self.dct_info, f, protocol=2)

class Trainer_oaf_alwaysMSE(Trainer_oaf):
	"""
	use MSE instead of KL for attention loss
	"""
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = MSELoss()
		nll_fr = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# fixed
			teacher_forcing_ratio = 0.5
			# linear schedule
			# teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		# import pdb; pdb.set_trace()
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		# print(decoder_samples.size(), decoder_samples[:5,:10])
		# print(tgt_ids.size(), tgt_ids[:5,:10])
		# pdb.set_trace()
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:])
			nll_fr.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
			nll_fr.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		# nll_fr.mul(1.0 - teacher_forcing_ratio)
		
		# Get KL loss
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		klloss.eval_batch_seq_with_mask(attn_hyp_fr, 
										attn_hyp.detach(), 
										non_padding_mask_tgt[:, 1:]
										)

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# addition
		# total_loss = loss.add(klloss)

		# oaf loss
		fr_percent = 1.0 - teacher_forcing_ratio
		total_loss = (1.0 - fr_percent) * loss.acc_loss + fr_percent * nll_fr.acc_loss + klloss.acc_loss

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resloss_fr = nll_fr.get_loss()
		resklloss = klloss.get_loss()
		
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resloss_fr, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss

class Trainer_oaf_alwaysKLsmooth(Trainer_oaf):
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()
		nll_fr = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# fixed
			teacher_forcing_ratio = 0.5
			# linear schedule
			# teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		# import pdb; pdb.set_trace()
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		# print(decoder_samples.size(), decoder_samples[:5,:10])
		# print(tgt_ids.size(), tgt_ids[:5,:10])
		# pdb.set_trace()
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:])
			nll_fr.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
			nll_fr.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		# nll_fr.mul(1.0 - teacher_forcing_ratio)
		
		# Get KL loss
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		klloss.eval_batch_seq_with_mask_smooth(attn_hyp_fr, 
										attn_hyp.detach(), 
										non_padding_mask_tgt[:, 1:]
										)

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# addition
		# total_loss = loss.add(klloss)

		# oaf loss
		fr_percent = 1.0 - teacher_forcing_ratio
		total_loss = (1.0 - fr_percent) * loss.acc_loss + fr_percent * nll_fr.acc_loss + klloss.acc_loss

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resloss_fr = nll_fr.get_loss()
		resklloss = klloss.get_loss()
		
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resloss_fr, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss

class Trainer_oaf_alwaysKL(Trainer_oaf):
	"""
	always use the alignment loss KL(alpha||alpha')
	"""
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()
		nll_fr = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# fixed
			# teacher_forcing_ratio = 0.5
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		# import pdb; pdb.set_trace()
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		# print(decoder_samples.size(), decoder_samples[:5,:10])
		# print(tgt_ids.size(), tgt_ids[:5,:10])
		# pdb.set_trace()
		attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs, att_scores=attn_hyp.detach())
		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:])
			nll_fr.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
			nll_fr.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		# nll_fr.mul(1.0 - teacher_forcing_ratio)
		
		# Get KL loss
		assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		klloss.eval_batch_seq_with_mask(attn_hyp_fr, 
										attn_hyp.detach(), 
										non_padding_mask_tgt[:, 1:]
										)

		# add coeff
		klloss.mul(self.attention_loss_coeff)

		# addition
		# total_loss = loss.add(klloss)

		# oaf loss
		fr_percent = 1.0 - teacher_forcing_ratio
		total_loss = (1.0 - fr_percent) * loss.acc_loss + fr_percent * nll_fr.acc_loss + klloss.acc_loss

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resloss_fr = nll_fr.get_loss()
		resklloss = klloss.get_loss()
		
		# print(resloss, resklloss, fr_percent)

		self._update_dct_info(step, resloss, resloss_fr, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss

class Trainer_oaf_noKL(Trainer_oaf):
	"""
	never use the alignment loss KL(alpha||alpha'), equivalent to oracle SS
	"""
	def _train_batch(self, src_ids, tgt_ids, model, step, total_steps, src_probs=None, attscores=None):
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()
		nll_fr = NLLLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# fixed
			# teacher_forcing_ratio = 0.5
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# NB: flag_stack_outputs==True
		# get 2 versions: tf model always uses ref history, af model uses tf-gen history
		model.attention_forcing = False
		decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
		decoder_outputs = torch.stack(decoder_outputs, dim=2)
		decoder_samples = torch.cat(ret_dict['sequence'], dim=1)
		decoder_samples = torch.cat([tgt_ids[:,0:1], decoder_samples], dim=1)
		# attn_hyp = torch.cat(ret_dict['attention_score'], dim=1)

		# model.attention_forcing = True
		decoder_outputs_fr, decoder_hidden_fr, ret_dict_fr = model(src_ids, decoder_samples, 
												is_training=True, 
												teacher_forcing_ratio=1.0,
												att_key_feats=src_probs)
												# att_key_feats=src_probs, att_scores=attn_hyp.detach())

		decoder_outputs_fr = torch.stack(decoder_outputs_fr, dim=2)
		# attn_hyp_fr = torch.cat(ret_dict_fr['attention_score'], dim=1)

		# Print out intermediate results
		# code removed for concise log

		# Get loss 
		# 2 versions, use both
		loss.reset()
		if not self.eval_with_mask:
			loss.eval_batch_seq(decoder_outputs, 
				tgt_ids[:, 1:])
			nll_fr.eval_batch_seq(decoder_outputs_fr, 
				tgt_ids[:, 1:])
		else:
			# .contiguous() is for safety / consistency
			loss.eval_batch_seq_with_mask(decoder_outputs, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
			nll_fr.eval_batch_seq_with_mask(decoder_outputs_fr, 
				tgt_ids[:, 1:], 
				non_padding_mask_tgt[:, 1:])
		# nll_fr.mul(1.0 - teacher_forcing_ratio)
		
		# # Get KL loss
		# assert self.attention_loss_coeff > 0, 'self.attention_loss_coeff > 0 required, but got {}'.format(self.attention_loss_coeff)
		# klloss.eval_batch_seq_with_mask(attn_hyp_fr, 
		# 								attn_hyp.detach(), 
		# 								non_padding_mask_tgt[:, 1:]
		# 								)

		# # add coeff
		# klloss.mul(self.attention_loss_coeff)

		# addition
		# total_loss = loss.add(klloss)

		# oaf loss
		fr_percent = 1.0 - teacher_forcing_ratio
		total_loss = (1.0 - fr_percent) * loss.acc_loss + fr_percent * nll_fr.acc_loss

		# Backward propagation
		model.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		resloss = loss.get_loss()
		resloss_fr = nll_fr.get_loss()
		resklloss = klloss.get_loss()
		
		# print(resloss, resklloss, fr_percent)
		self._update_dct_info(step, resloss, resloss_fr, resklloss, fr_percent)
		self._print_seqs(model, step, src_ids, tgt_ids, ret_dict, ret_dict_fr=ret_dict_fr)

		return resloss, resklloss	



class Trainer_dual(object):

	"""
		simultaneously train tf/af model
		separately update; only usng att kl to relate the two
	"""

	def __init__(self, expt_dir='experiment', 
		load_dir=None,
		batch_size=64, 
		random_seed=None,
		checkpoint_every=100, 
		print_every=100, 
		use_gpu=False,
		learning_rate=0.001, 
		max_grad_norm=1.0,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=1.0,
		attention_loss_coeff=1.0):

		self.random_seed = random_seed
		if random_seed is not None:
			set_global_seeds(random_seed)

		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.use_gpu = use_gpu
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.attention_loss_coeff = attention_loss_coeff

		self.optimizer_tf = None
		self.optimizer_af = None
		self.learning_rate = learning_rate
		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir_tf = os.path.join(expt_dir, 'tf')
		self.expt_dir_af = os.path.join(expt_dir, 'af')
		if not os.path.exists(self.expt_dir_tf):
			os.makedirs(self.expt_dir_tf)
		if not os.path.exists(self.expt_dir_af):
			os.makedirs(self.expt_dir_af)
		if type(load_dir) != type(None):
			self.load_dir_tf = os.path.join(load_dir, 'tf')
			self.load_dir_af = os.path.join(load_dir, 'af')
		self.expt_dir = expt_dir
		self.load_dir = load_dir

		self.batch_size = batch_size
		self.logger = logging.getLogger(__name__)
		# self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)


	def _train_batch(self, src_ids, tgt_ids, model_tf, model_af, step, total_steps):
		
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss_tf = NLLLoss()
		loss_af = NLLLoss()
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			# use self.teacher_forcing_ratio as the starting point
			progress = 1.0 * step / total_steps
			if progress < 0.4:
				teacher_forcing_ratio = self.teacher_forcing_ratio
			elif progress < 0.8:
				teacher_forcing_ratio = 1.0 * self.teacher_forcing_ratio / 2
			elif progress < 0.9:
				teacher_forcing_ratio = 0.3
			else:
				teacher_forcing_ratio = 0.0 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		decoder_outputs_tf, decoder_hidden_tf, ret_dict_tf = model_tf(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=teacher_forcing_ratio,
												att_key_feats=None, att_scores=None)

		# get ref attention score from tf model
		attn_tf = ret_dict_tf['attention_score']
		decoder_outputs_af, decoder_hidden_af, ret_dict_af = model_af(src_ids, tgt_ids, 
												is_training=True, 
												teacher_forcing_ratio=0.0,
												att_key_feats=None, att_scores=attn_tf)
		# print(len(decoder_outputs))	# max_seq_len - 1

		# Print out intermediate results
		if step % self.checkpoint_every == 0 or step == 1:
			seqlist_tf = ret_dict_tf['sequence']
			seqlist_af = ret_dict_af['sequence']
			# convert to words
			srcwords = _convert_to_words_batchfirst(src_ids, model_tf.id2word_enc)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], model_tf.id2word_dec)			
			seqwords_tf = _convert_to_words(seqlist_tf, model_tf.id2word_dec)
			seqwords_af = _convert_to_words(seqlist_af, model_af.id2word_dec)

			print('---step_res---')
			for i in range(3):
				print('---{}---'.format(i))
				outsrc = 	'SRC:    {}\n'.format(' '.join(srcwords[i])).encode('utf-8')
				outref = 	'REF:    {}\n'.format(' '.join(refwords[i])).encode('utf-8')
				outline1 = 	'OUT-TF: {}\n'.format(' '.join(seqwords_tf[i])).encode('utf-8')
				outline2 = 	'OUT-AF: {}\n'.format(' '.join(seqwords_af[i])).encode('utf-8')
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline1)
				sys.stdout.buffer.write(outline2)
			print('----------------')
			# input('...')
			sys.stdout.flush()

		# Get NLL loss 
		loss_tf.reset()
		for step_tf, step_output_tf in enumerate(decoder_outputs_tf):
			if not self.eval_with_mask:
				loss_tf.eval_batch(step_output_tf.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step_tf+1])
			else:
				loss_tf.eval_batch_with_mask(step_output_tf.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step_tf+1], non_padding_mask_tgt[:, step_tf+1])
		loss_af.reset()
		for step_af, step_output_af in enumerate(decoder_outputs_af):
			if not self.eval_with_mask:
				loss_af.eval_batch(step_output_af.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step_af+1])
			else:
				loss_af.eval_batch_with_mask(step_output_af.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step_af+1], non_padding_mask_tgt[:, step_af+1])

		# Get KL loss
		attn_ref = ret_dict_af['attention_ref']
		attn_hyp = ret_dict_af['attention_score']
		for idx in range(len(attn_hyp)):
			klloss.eval_batch_with_mask_v2(attn_hyp[idx].contiguous(), attn_ref[idx].contiguous(), non_padding_mask_tgt[:, idx+1])
		klloss.mul(self.attention_loss_coeff)

		# Backward propagation
		model_tf.zero_grad()
		model_af.zero_grad()
		
		# --- debug block: can add after backward/step ---
		# print('initial: TF')
		# input('.........')
		# for name, param in model_tf.named_parameters():
		# 	if 'out.weight' in name:
		# 		print('grad {}:{}'.format(name, param.grad))
		# 		print('val {}:{}'.format(name, param.data))		

		loss_tf.backward()
		self.optimizer_tf.step()		
		loss_af.backward(retain_graph=True)
		self.optimizer_af.step()
		klloss.backward()
		self.optimizer_af.step()

		resloss_tf = loss_tf.get_loss()
		resloss_af = loss_af.get_loss()
		resklloss = klloss.get_loss()	

		return resloss_tf, resloss_af, resklloss


	def _evaluate_batches(self, model, batches, dataset):

		model.eval()

		loss = self.loss
		loss.reset()
		match = 0
		total = 0

		out_count = 0
		with torch.no_grad():
			for batch in batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)

				decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
														is_training=False,
														att_key_feats=src_probs)

				# Evaluation
				seqlist = other['sequence']
				for step, step_output in enumerate(decoder_outputs):
					target = tgt_ids[:, step+1]
					non_padding = target.ne(PAD)

					if not self.eval_with_mask:
						loss.eval_batch(step_output.view(tgt_ids.size(0), -1), target)
					else:
						loss.eval_batch_with_mask(step_output.view(tgt_ids.size(0), -1), target, non_padding)

					correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()
				
				if out_count < 3:
					refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], dataset.tgt_id2word)
					seqwords = _convert_to_words(seqlist, dataset.tgt_id2word)
					outref = 'REF: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
					outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
					sys.stdout.buffer.write(outref)
					sys.stdout.buffer.write(outline)
					out_count += 1


		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		resloss = loss.get_loss()
		torch.cuda.empty_cache()

		return resloss, accuracy
		

	def _train_epoches(self, train_set, model_tf, model_af, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total_tf = 0  # Reset every print_every
		epoch_loss_total_tf = 0  # Reset every epoch
		prev_acc_tf = 0.0
		prev_epoch_acc_tf = 0.0
		ckpt_tf = None

		print_loss_total_af = 0  # Reset every print_every
		epoch_loss_total_af = 0  # Reset every epoch
		prev_acc_af = 0.0
		prev_epoch_acc_af = 0.0
		ckpt_af = None

		print_klloss_total = 0  # Reset every print_every
		epoch_klloss_total = 0  # Reset every epoch
		step = start_step
		step_elapsed = 0

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			# ----------construct batches-----------
			# allow re-shuffling of data
			print('--- construct train set ---')
			train_batches, vocab_size = train_set.construct_batches(is_train=True)

			if dev_set is not None:
				print('--- construct dev set ---')
				dev_batches, vocab_size = dev_set.construct_batches(is_train=False)

			# --------print info for each epoch----------
			steps_per_epoch = len(train_batches)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))

			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			# self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model_tf.train(True)
			model_af.train(True)
			for batch in train_batches:

				# update macro count
				step += 1
				step_elapsed += 1

				# load data
				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check src tgt pair ---')
					log_msgs = check_srctgt(src_ids, tgt_ids, train_set.src_id2word, train_set.tgt_id2word)
					for log_msg in log_msgs:
						sys.stdout.buffer.write(log_msg)
						# print(log_msg)

				# convert variable to tensor
				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)
				# print(s rc_probs.size())

				# Get loss
				loss_tf, loss_af, klloss = self._train_batch(src_ids, tgt_ids, model_tf, model_af, step, total_steps)

				print_loss_total_tf += loss_tf
				epoch_loss_total_tf += loss_tf
				print_loss_total_af += loss_af
				epoch_loss_total_af += loss_af
				print_klloss_total += klloss
				epoch_klloss_total += klloss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_avg_tf = print_loss_total_tf / self.print_every
					print_loss_total_tf = 0
					print_loss_avg_af = print_loss_total_af / self.print_every
					print_loss_total_af = 0
					print_klloss_avg = print_klloss_total / self.print_every
					print_klloss_total = 0
					log_msg = 'Progress: %d%%, Train TFloss: %.4f, AFloss: %.4f, att klloss: %.4f,' % (
								step / total_steps * 100,
								print_loss_avg_tf,
								print_loss_avg_af,
								print_klloss_avg)
					# print(log_msg)
					log.info(log_msg)
					# self.writer.add_scalar('train_loss_tf', print_loss_avg_tf, global_step=step)
					# self.writer.add_scalar('train_loss_af', print_loss_avg_af, global_step=step)
					# self.writer.add_scalar('train_klloss', print_klloss_avg, global_step=step)

				# Checkpoint
				# if step % self.checkpoint_every == 0 or step == total_steps or step == 1:
				if step == 1:	
					ckpt_tf = Checkpoint(model=model_tf,
							   optimizer=self.optimizer_tf,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					ckpt_af = Checkpoint(model=model_af,
							   optimizer=self.optimizer_af,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					# save criteria
					if dev_set is not None:
						dev_loss_tf, accuracy_tf = self._evaluate_batches(model_tf, dev_batches, dev_set)
						dev_loss_af, accuracy_af = self._evaluate_batches(model_af, dev_batches, dev_set)
						print('TF dev loss: {}, accuracy: {}'.format(dev_loss_tf, accuracy_tf))
						print('AF dev loss: {}, accuracy: {}'.format(dev_loss_af, accuracy_af))
						model_tf.train(mode=True)
						model_af.train(mode=True)

						if prev_acc_tf < accuracy_tf:
							# save the best model
							saved_path_tf = ckpt_tf.save(self.expt_dir_tf)
							print('saving at {} ... '.format(saved_path_tf))
							prev_acc_tf = accuracy_tf
							# keep best 5 models
							ckpt_tf.rm_old(self.expt_dir_tf, keep_num=5)

						if prev_acc_af < accuracy_af:
							# save the best model
							saved_path_af = ckpt_af.save(self.expt_dir_af)
							print('saving at {} ... '.format(saved_path_af))
							prev_acc_af = accuracy_af
							# keep best 5 models
							ckpt_af.rm_old(self.expt_dir_af, keep_num=5)

					else:
						saved_path_tf = ckpt_tf.save(self.expt_dir_tf)
						print('saving tf model at {} ... '.format(saved_path_tf))
						ckpt_tf.rm_old(self.expt_dir_tf, keep_num=2)
						saved_path_af = ckpt_af.save(self.expt_dir_af)
						print('saving af model at {} ... '.format(saved_path_af))
						ckpt_af.rm_old(self.expt_dir_af, keep_num=2)
				
				sys.stdout.flush()

			if step_elapsed == 0: continue
			epoch_loss_avg_tf = epoch_loss_total_tf / min(steps_per_epoch, step - start_step)
			epoch_loss_total_tf = 0
			epoch_loss_avg_af = epoch_loss_total_af / min(steps_per_epoch, step - start_step)
			epoch_loss_total_af = 0
			epoch_klloss_avg = epoch_klloss_total / min(steps_per_epoch, step - start_step)
			epoch_klloss_total = 0
			log_msg = "Finished epoch %d: Train TFloss: %.4f AFloss: %.4f att klloss: %.4f" \
								% (epoch, epoch_loss_avg_tf, epoch_loss_avg_af, epoch_klloss_avg)

			# ********************** [finish 1 epoch: eval on dev] ***********************	
			if dev_set is not None:
				# stricter criteria to save if dev set is available - only save when performance improves on dev set
				dev_loss_tf, epoch_accuracy_tf = self._evaluate_batches(model_tf, dev_batches, dev_set)
				dev_loss_af, epoch_accuracy_af = self._evaluate_batches(model_af, dev_batches, dev_set)
				self.optimizer_tf.update(dev_loss_tf, epoch)
				self.optimizer_af.update(dev_loss_af, epoch)
				# self.writer.add_scalar('dev_loss_tf', dev_loss_tf, global_step=step)
				# self.writer.add_scalar('dev_acc_tf', accuracy_tf, global_step=step)
				# self.writer.add_scalar('dev_loss_af', dev_loss_af, global_step=step)
				# self.writer.add_scalar('dev_acc_af', accuracy_af, global_step=step)
				log_msg += ", Dev TFloss %.4f, Accuracy: %.4f, AFloss %.4f, Accuracy: %.4f" \
								% (dev_loss_tf, epoch_accuracy_tf, dev_loss_af, epoch_accuracy_af)
				model.train(mode=True)
				if prev_epoch_acc_tf < epoch_accuracy_tf:
					ckpt_tf = Checkpoint(model=model_tf,
								   optimizer=self.optimizer_tf,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_tgt)

					saved_path_tf = ckpt_tf.save_epoch(self.expt_dir_tf, epoch)
					print('saving teacher forcing model at {} ... '.format(saved_path_tf))
					prev_epoch_acc_tf = epoch_accuracy_tf			
				if prev_epoch_acc_af < epoch_accuracy_af:
					ckpt_af = Checkpoint(model=model_af,
								   optimizer=self.optimizer_af,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_tgt)

					saved_path_af = ckpt_af.save_epoch(self.expt_dir_af, epoch)
					print('saving attention forcing model at {} ... '.format(saved_path_af))
					prev_epoch_acc_af = epoch_accuracy_af

			else:
				# save after finishing one epoch
				ckpt_tf = Checkpoint(model=model_tf,
							   optimizer=self.optimizer_tf,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
				
				ckpt_af = Checkpoint(model=model_af,
							   optimizer=self.optimizer_af,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)

				saved_path_tf = ckpt_tf.save_epoch(self.expt_dir_tf, epoch)
				print('saving teacher forcing model at {} ... '.format(saved_path_tf))			
				saved_path_af = ckpt_af.save_epoch(self.expt_dir_af, epoch)
				print('saving attention forcing model at {} ... '.format(saved_path_af))			

			log.info('\n')
			log.info(log_msg)


	def train(self, train_set, model_tf, model_af, num_epochs=5, resume=False, dev_set=None):

		""" 
			Run training for a given model.
			Args:
				train_set: Dataset
				dev_set: Dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run (default 5)
				resume(bool, optional): resume training with the latest checkpoint, (default False)
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
				   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
				
			Returns:
				model (seq2seq.models): trained model.
		"""

		torch.cuda.empty_cache()
		self.resume = resume
		if resume:

			# ------------------------------------------------------------------------------------
			# -------------------------- teacher forcing model resume ... 
			latest_checkpoint_path_tf = Checkpoint.get_latest_epoch_checkpoint(self.load_dir_tf)
			print('resuming teacher forcing model {} ...'.format(latest_checkpoint_path_tf))
			resume_checkpoint_tf = Checkpoint.load(latest_checkpoint_path_tf)
			model_tf = resume_checkpoint_tf.model
			self.optimizer_tf = resume_checkpoint_tf.optimizer

			# check var
			model_tf.set_var('attention_forcing', False)
			model_tf.set_var('debug_count', 0)
			model_tf.reset_use_gpu(self.use_gpu)
			model_tf.to(device)
			print('attention forcing: {}'.format(model_tf.attention_forcing))
			print('use gpu: {}'.format(model_tf.use_gpu))

			# A walk around to set optimizing parameters properly
			resume_optim_tf = self.optimizer_tf.optimizer
			defaults_tf = resume_optim_tf.param_groups[0]
			defaults_tf.pop('params', None)
			defaults_tf.pop('initial_lr', None)
			self.optimizer_tf.optimizer = resume_optim_tf.__class__(model_tf.parameters(), **defaults_tf)

			# ------------------------------------------------------------------------------------
			# ----------------------------- attention forcing model resume ... 
			latest_checkpoint_path_af = Checkpoint.get_latest_epoch_checkpoint(self.load_dir_af)
			print('resuming attention forcing model {} ...'.format(latest_checkpoint_path_af))
			resume_checkpoint_af = Checkpoint.load(latest_checkpoint_path_af)
			model_af = resume_checkpoint_af.model
			self.optimizer_af = resume_checkpoint_af.optimizer

			# check var
			model_af.set_var('attention_forcing', True)
			model_af.set_var('debug_count', 0)
			model_af.reset_use_gpu(self.use_gpu)
			model_af.to(device)
			print('attention forcing: {}'.format(model_af.attention_forcing))
			print('use gpu: {}'.format(model_af.use_gpu))

			# A walk around to set optimizing parameters properly
			resume_optim_af = self.optimizer_af.optimizer
			defaults_af = resume_optim_af.param_groups[0]
			defaults_af.pop('params', None)
			defaults_af.pop('initial_lr', None)
			self.optimizer_af.optimizer = resume_optim_af.__class__(model_af.parameters(), **defaults_af)

			# ------------------------------------------
			start_epoch = resume_checkpoint_af.epoch
			step = resume_checkpoint_af.step

		else:
			start_epoch = 1
			step = 0

			log = self.logger.info("------ teacher forcing model ------ ")
			for name, param in model_tf.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))
			log = self.logger.info("------ attention forcing model ------ ")
			for name, param in model_af.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))
				# check embedder init
				# if 'embedder' in name:
				# 	print('{}:{}'.format(name, param[5]))

			optimizer_tf = Optimizer(torch.optim.Adam(model_tf.parameters(), 
						lr=self.learning_rate), max_grad_norm=self.max_grad_norm) # 5 -> 1
			optimizer_af = Optimizer(torch.optim.Adam(model_af.parameters(), 
						lr=self.learning_rate), max_grad_norm=self.max_grad_norm) # 5 -> 1
			self.optimizer_tf = optimizer_tf # teacher forcing
			self.optimizer_af = optimizer_af # attention forcing

		self.logger.info("TF Optimizer: %s, Scheduler: %s" % (self.optimizer_tf.optimizer, self.optimizer_tf.scheduler))
		self.logger.info("AF Optimizer: %s, Scheduler: %s" % (self.optimizer_af.optimizer, self.optimizer_af.scheduler))

		self._train_epoches(train_set, model_tf, model_af, num_epochs, start_epoch, step, dev_set=dev_set)
		
		return model_tf, model_af


class Trainer_afdynamic(object):

	"""
		load pretrained tf model
		use tf to generate ref att
		use af to generate back history (word prediction)
		only update af param
	"""

	def __init__(self, expt_dir='experiment', 
		load_dir=None,
		load_tf_dir=None,
		loss=NLLLoss(), 
		batch_size=64, 
		random_seed=None,
		checkpoint_every=100, 
		print_every=100, 
		use_gpu=False,
		learning_rate=0.001, 
		max_grad_norm=1.0,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=1.0,
		attention_loss_coeff=1.0,
		attention_forcing=False):

		self.random_seed = random_seed
		if random_seed is not None:
			set_global_seeds(random_seed)

		self.loss = loss
		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.use_gpu = use_gpu
		self.learning_rate = learning_rate
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.attention_loss_coeff = attention_loss_coeff
		self.attention_forcing = attention_forcing

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir
		self.load_tf_dir = load_tf_dir

		self.batch_size = batch_size
		self.logger = logging.getLogger(__name__)
		# self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)
		print('scheduled sampling: {}'.format(self.scheduled_sampling))


	def _evaluate_batches(self, model, batches, dataset):

		model.eval()

		loss = self.loss
		loss.reset()
		match = 0
		total = 0

		out_count = 0
		with torch.no_grad():
			for batch in batches:

				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)

				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)

				decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
														is_training=False,
														att_key_feats=src_probs)

				# Evaluation
				seqlist = other['sequence']
				for step, step_output in enumerate(decoder_outputs):
					target = tgt_ids[:, step+1]
					non_padding = target.ne(PAD)

					if not self.eval_with_mask:
						loss.eval_batch(step_output.view(tgt_ids.size(0), -1), target)
					else:
						loss.eval_batch_with_mask(step_output.view(tgt_ids.size(0), -1), target, non_padding)

					correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()
				
				if out_count < 3:
					refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], dataset.tgt_id2word)
					seqwords = _convert_to_words(seqlist, dataset.tgt_id2word)
					outref = 'REF: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
					outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
					sys.stdout.buffer.write(outref)
					sys.stdout.buffer.write(outline)
					out_count += 1


		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		resloss = loss.get_loss()
		torch.cuda.empty_cache()

		return resloss, accuracy
		

	def _forward_aftf(self, model_tf, model_af, teacher_forcing_ratio, src, tgt=None, hidden=None):

		"""
			similar purpose to forward function in recurrent.py
			but with forward step using af/tf as such:
					<BOS> -> [TF] -> refatt -> [AF] -> <w1> 
				-> 	<w1> -> [TF] -> refatt -> [AF] -> <w2> etc....
			i.e. for each step:
				dynamically generate back history of words using [AF]
				dynamically generate ref attention using [TF] 
				foward in recurrent.py only allows one models to be used 
			note:
				[TF] freezed, not updating
				[AF] target model, to be updated
		"""

		"""
			Args:
				src: list of src word_ids [batch_size, max_seq_len, word_ids]
				tgt: list of tgt word_ids
				hidden: initial hidden state
			Returns:
				decoder_outputs: list of step_output - log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				ret_dict
		"""

		if self.use_gpu and torch.cuda.is_available():
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
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		for idx in range(max_seq_len - 1):

			# 5.1 gen refatt: [TF] w1 -> tf -> att_tf (-> tf -> w2_tf)
			predicted_softmax_tf, dec_hidden_tf, step_attn_tf, c_out_tf, cell_value_tf = \
				model_tf.forward_step(att_keys_tf, att_vals_tf, tgt_chunk_tf, cell_value_tf, dec_hidden_tf, mask_src, prev_c_tf, use_gpu=self.use_gpu)
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
					tgt_chunk_af, cell_value_af, dec_hidden_af, mask_src, prev_c_af, att_ref=step_attn_ref_detach, use_gpu=self.use_gpu)
			step_output_af = predicted_softmax_af.squeeze(1)
			symbols_af, lengths_af, sequence_symbols = model_tf.forward_decode(idx, step_output_af, lengths_af, sequence_symbols)
			prev_c_af = c_out_af
			# import pdb; pdb.set_trace()

			# 5.4 store var for af model
			ret_dict[KEY_ATTN_SCORE].append(step_attn_af)
			decoder_outputs.append(step_output_af)

			# 5.5 set w2 as w2_af 
			if use_teacher_forcing:
				tgt_chunk_af = emb_tgt_af[:, idx+1].unsqueeze(1)
				tgt_chunk_tf = emb_tgt_tf[:, idx+1].unsqueeze(1)
			else:
				tgt_chunk_af = model_af.embedder_dec(symbols_af)
				tgt_chunk_tf = model_tf.embedder_dec(symbols_af)
			# if idx < 5:
			# 	import pdb; pdb.set_trace()

		# print('...')
		ret_dict[KEY_SEQUENCE] = sequence_symbols
		ret_dict[KEY_LENGTH] = lengths_af.tolist()

		return decoder_outputs, dec_hidden_af, ret_dict		


	def _train_batch(self, src_ids, tgt_ids, model_tf, model_af, step, total_steps, src_probs=None, attscores=None):
		
		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>
			(optional)
				src_probs 		=     p1 p2 p3 0    0     ...
				attscores 		n * [31*32] numpy array 
			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# define loss
		loss = self.loss
		klloss = KLDivLoss()

		# scheduled sampling
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			# linear schedule
			teacher_forcing_ratio = 1.0 - progress 

		# get padding mask
		non_padding_mask_src = src_ids.data.ne(PAD)
		non_padding_mask_tgt = tgt_ids.data.ne(PAD)

		# Forward propagation
		# print(teacher_forcing_ratio)
		decoder_outputs, decoder_hidden, ret_dict = self._forward_aftf(model_tf, model_af, teacher_forcing_ratio, src_ids, tgt_ids)
		# print(len(decoder_outputs))	# max_seq_len - 1

		# Print out intermediate results
		if step % self.checkpoint_every == 0:
			seqlist = ret_dict['sequence']
			# convert to words
			srcwords = _convert_to_words_batchfirst(src_ids, model_af.id2word_enc)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], model_af.id2word_dec)			
			seqwords = _convert_to_words(seqlist, model_af.id2word_dec)

			print('---step_res---')
			for i in range(3):
				print('---{}---'.format(i))
				outsrc = 'SRC: {}\n'.format(' '.join(srcwords[i])).encode('utf-8')
				outref = 'REF: {}\n'.format(' '.join(refwords[i])).encode('utf-8')
				outline = 'OUT: {}\n'.format(' '.join(seqwords[i])).encode('utf-8')
				sys.stdout.buffer.write(outsrc)
				sys.stdout.buffer.write(outref)
				sys.stdout.buffer.write(outline)
			print('----------------')
			# input('...')
			sys.stdout.flush()

		# Get loss 
		loss.reset()
		for step, step_output in enumerate(decoder_outputs):
			# iterate over seq_len
			if not self.eval_with_mask:
				# print('Train with penalty on mask')
				loss.eval_batch(step_output.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step+1])
			else:
				# print('Train without penalty on mask')
				loss.eval_batch_with_mask(step_output.contiguous()\
					.view(self.batch_size, -1), tgt_ids[:, step+1], non_padding_mask_tgt[:, step+1])
			# print(loss.acc_loss)
			# input('...')

		if self.attention_loss_coeff > 0:

			# Get KL loss
			attn_hyp = ret_dict['attention_score']
			attn_ref = ret_dict['attention_ref']
			for idx in range(len(attn_hyp)):
				klloss.eval_batch_with_mask_v2(attn_hyp[idx].contiguous(), attn_ref[idx].contiguous(), non_padding_mask_tgt[:, idx+1])
				# print(klloss.acc_loss)
				# input('...')

			# add coeff
			klloss.mul(self.attention_loss_coeff)

			# addition
			total_loss = loss.add(klloss)

			# Backward propagation
			model_af.zero_grad()
			total_loss.backward()
			resklloss = klloss.get_loss()	

		else:

			# no attention forcing
			model_af.zero_grad()
			loss.backward()
			resklloss = 0

		self.optimizer.step()
		resloss = loss.get_loss()

		return resloss, resklloss


	def _train_epoches(self, train_set, model_tf, model_af, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total = 0  # Reset every print_every
		epoch_loss_total = 0  # Reset every epoch
		print_klloss_total = 0  # Reset every print_every
		epoch_klloss_total = 0  # Reset every epoch

		step = start_step
		step_elapsed = 0
		ckpt = None
		prev_acc = 0.0
		prev_epoch_acc = 0.0

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			# ----------construct batches-----------
			# allow re-shuffling of data
			if type(train_set.attscore_path) != type(None):
				print('--- construct train set (with attscore) ---')
				train_batches, vocab_size = train_set.construct_batches_with_attscore(is_train=True)
			else:
				print('--- construct train set ---')
				train_batches, vocab_size = train_set.construct_batches(is_train=True)

			if dev_set is not None:
				if type(dev_set.attscore_path) != type(None):
					print('--- construct dev set (with attscore) ---')
					dev_batches, vocab_size = dev_set.construct_batches_with_attscore(is_train=False)
				else:
					print('--- construct dev set ---')
					dev_batches, vocab_size = dev_set.construct_batches(is_train=False)


			# --------print info for each epoch----------
			steps_per_epoch = len(train_batches)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))


			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			# self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model_tf.train(False)
			model_af.train(True)
			for batch in train_batches:

				# print(step)

				# update macro count
				step += 1
				step_elapsed += 1

				# load data
				src_ids = batch['src_word_ids']
				src_lengths = batch['src_sentence_lengths']
				tgt_ids = batch['tgt_word_ids']
				tgt_lengths = batch['tgt_sentence_lengths']
				src_probs = None
				attscores = None
				if 'src_ddfd_probs' in batch:
					src_probs =  batch['src_ddfd_probs']
					src_probs = _convert_to_tensor(src_probs, self.use_gpu).unsqueeze(2)
				if 'attscores' in batch:
					attscores = batch['attscores'] #list of numpy arrays
					attscores = _convert_to_tensor(attscores, self.use_gpu) #n*31*32

				# sanity check src-tgt pair
				if step == 1:
					print('--- Check src tgt pair ---')
					log_msgs = check_srctgt(src_ids, tgt_ids, train_set.src_id2word, train_set.tgt_id2word)
					for log_msg in log_msgs:
						sys.stdout.buffer.write(log_msg)
						# print(log_msg)

				# convert variable to tensor
				src_ids = _convert_to_tensor(src_ids, self.use_gpu)
				tgt_ids = _convert_to_tensor(tgt_ids, self.use_gpu)
				# print(s rc_probs.size())

				# Get loss
				loss, klloss = self._train_batch(src_ids, tgt_ids, model_tf, model_af, step, total_steps, 
												src_probs=src_probs, attscores=attscores)

				print_loss_total += loss
				epoch_loss_total += loss
				print_klloss_total += klloss
				epoch_klloss_total += klloss
				# import pdb; pdb.set_trace()

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0
					print_klloss_avg = print_klloss_total / self.print_every
					print_klloss_total = 0
					log_msg = 'Progress: %d%%, Train %s: %.4f, att klloss: %.4f,' % (
								step / total_steps * 100,
								self.loss.name,
								print_loss_avg,
								print_klloss_avg)
					# print(log_msg)
					log.info(log_msg)
					# self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)
					# self.writer.add_scalar('train_klloss', print_klloss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps or step == 1:
					# if step == 1:	
					ckpt = Checkpoint(model=model_af,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					# save criteria
					if dev_set is not None:
						dev_loss, accuracy = self._evaluate_batches(model_af, dev_batches, dev_set)
						print('dev loss: {}, accuracy: {}'.format(dev_loss, accuracy))
						model_af.train(mode=True)

						if prev_acc < accuracy:
						# if True:	
							# save the best model_af
							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))
							prev_acc = accuracy
							# keep best 5 models
							ckpt.rm_old(self.expt_dir, keep_num=5)

						# else:
						# 	# load last best model - disable [this froze the training..]
						# 	latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
						# 	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
						# 	model = resume_checkpoint.model

					else:
						saved_path = ckpt.save(self.expt_dir)
						print('saving at {} ... '.format(saved_path))
						# keep last 2 models
						ckpt.rm_old(self.expt_dir, keep_num=2)

					# save the last ckpt
					# if step == total_steps:
					# 	saved_path = ckpt.save(self.expt_dir)
					# 	print('saving at {} ... '.format(saved_path))
				
				sys.stdout.flush()

			if step_elapsed == 0: continue
			epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
			epoch_loss_total = 0
			epoch_klloss_avg = epoch_klloss_total / min(steps_per_epoch, step - start_step)
			epoch_klloss_total = 0
			log_msg = "Finished epoch %d: Train %s: %.4f att klloss: %.4f" % (epoch, self.loss.name, epoch_loss_avg, epoch_klloss_avg)

			# ********************** [finish 1 epoch: eval on dev] ***********************	
			if dev_set is not None:
				# stricter criteria to save if dev set is available - only save when performance improves on dev set
				dev_loss, epoch_accuracy = self._evaluate_batches(model_af, dev_batches, dev_set)
				self.optimizer.update(dev_loss, epoch)
				# self.writer.add_scalar('dev_loss', dev_loss, global_step=step)
				# self.writer.add_scalar('dev_acc', accuracy, global_step=step)
				log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, epoch_accuracy)
				model_af.train(mode=True)
				if prev_epoch_acc < epoch_accuracy:
					# save after finishing one epoch
					if ckpt is None:
						ckpt = Checkpoint(model=model_af,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_tgt)

					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					print('saving at {} ... '.format(saved_path))
					prev_epoch_acc = epoch_accuracy			
			else:
				self.optimizer.update(epoch_loss_avg, epoch)
				# save after finishing one epoch
				if ckpt is None:
					ckpt = Checkpoint(model=model_af,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)

				saved_path = ckpt.save_epoch(self.expt_dir, epoch)
				print('saving at {} ... '.format(saved_path))			

			log.info('\n')
			log.info(log_msg)


	def train(self, train_set, model_af, num_epochs=5, resume=False, optimizer=None, dev_set=None):

		""" 
			Run training for af model (with tf model generating ref att)
			Args:
				train_set: Dataset
				dev_set: Dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run (default 5)
				resume(bool, optional): resume training with the latest checkpoint, (default False)
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
				   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
				
			Returns:
				model (seq2seq.models): trained model.
		"""

		torch.cuda.empty_cache()

		dropout_rate = model_af.dropout_rate
		print('dropout {}'.format(dropout_rate))

		# load tf model
		latest_checkpoint_path = self.load_tf_dir
		resume_checkpoint_tf = Checkpoint.load(latest_checkpoint_path)
		model_tf = resume_checkpoint_tf.model.to(device)
		model_tf.reset_dropout(dropout_rate)
		model_tf.reset_use_gpu(self.use_gpu)
		model_tf.reset_batch_size(self.batch_size)

		print('TF Model dir: {}'.format(latest_checkpoint_path))
		print('TF Model laoded')

		# deal with af model init
		self.resume = resume
		if resume:
			# latest_checkpoint_path = Checkpoint.get_latest_epoch_checkpoint(self.load_dir)
			latest_checkpoint_path = self.load_dir
			print('resuming {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model_af = resume_checkpoint.model.to(device)
			self.optimizer = resume_checkpoint.optimizer

			# check var
			model_af.set_var('attention_forcing', self.attention_forcing)
			model_af.set_var('debug_count', 0)
			model_af.reset_dropout(dropout_rate)
			model_af.reset_use_gpu(self.use_gpu)
			model_af.reset_batch_size(self.batch_size)
			print('attention forcing: {}'.format(model_af.attention_forcing))
			print('use gpu: {}'.format(model_af.use_gpu))
			if self.use_gpu:
				model_af = model_af.cuda()
			else:
				model_af = model_af.cpu()

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model_af.parameters(), **defaults)

			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

			RESTART = True
			if RESTART:
				start_epoch = 1
				step = 0

		else:
			start_epoch = 1
			step = 0

			for name, param in model_af.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))
				# check embedder init
				# if 'embedder' in name:
				# 	print('{}:{}'.format(name, param[5]))

			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(model_af.parameters(), 
							lr=self.learning_rate), max_grad_norm=self.max_grad_norm) # 5 -> 1

				# set scheduler
				# optimizer.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'min'))

			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(train_set, model_tf, model_af, num_epochs, start_epoch, step, dev_set=dev_set)
		
		return model_af




def main():
	# load config
	parser = argparse.ArgumentParser(description='PyTorch Seq2Seq NMT Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# fix random seed
	if config['random_seed']!=666:
		set_global_seeds(config['random_seed'])

	# resume or not
	if config['load']:
		resume = True
		print('resuming {} ...'.format(config['load']))
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg') 
	else:
		resume = False
		config_save_dir = os.path.join(config['save'], 'model.cfg') 
	save_config(config, config_save_dir)

	# load train set
	train_path_src = config['train_path_src']
	train_path_tgt = config['train_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	train_attscore_path = config['train_attscore_path']

	train_set = Dataset(train_path_src, train_path_tgt,
						path_vocab_src, path_vocab_tgt,
						attscore_path=train_attscore_path,
						max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
						use_gpu=config['use_gpu'], use_type=config['use_type'])

	vocab_size_enc = len(train_set.vocab_src)
	vocab_size_dec = len(train_set.vocab_tgt)

	# load dev set
	if config['dev_path_src'] and config['dev_path_tgt']: 
		dev_path_src = config['dev_path_src']
		dev_path_tgt = config['dev_path_tgt']
		dev_attscore_path = config['dev_attscore_path']
		dev_set = Dataset(dev_path_src, dev_path_tgt,
						path_vocab_src, path_vocab_tgt,
						attscore_path=dev_attscore_path,
						max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
						use_gpu=config['use_gpu'])
	else:
		dev_set = None

	# import pdb; pdb.set_trace()
	
	if config['train_mode'] == 'multi':

		""" train either tf or af model """

		# make consistent
		if config['attention_forcing'] == False:
			config['attention_loss_coeff'] = 0.0

		# construct model
		seq2seq_dd = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=config['attention_forcing']).to(device)

		# contruct trainer
		t = Trainer(expt_dir=config['save'], 
						load_dir=config['load'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						attention_forcing=config['attention_forcing'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm']
						)

		# run training
		seq2seq_dd = t.train(train_set, seq2seq_dd, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)

	if config['train_mode'] == 'aaf_base':

		""" train either tf or af model -> train af model with AAF """

		# make consistent
		if config['attention_forcing'] == False:
			config['attention_loss_coeff'] = 0.0

		# construct model
		seq2seq_dd = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=config['attention_forcing']).to(device)

		# contruct trainer
		t = Trainer_aaf_base(expt_dir=config['save'], 
						load_dir=config['load'],
						load_tf_dir=config['load_tf'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						attention_forcing=config['attention_forcing'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm']
						)

		# run training
		seq2seq_dd = t.train(train_set, seq2seq_dd, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)


	if config['train_mode'] == 'aaf':
		# import pdb; pdb.set_trace()

		""" train either tf or af model """

		# make consistent
		if config['attention_forcing'] == False:
			config['attention_loss_coeff'] = 0.0

		# construct model
		seq2seq_dd = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=config['attention_forcing'],
								flag_stack_outputs=False).to(device)

		# contruct trainer
		t = Trainer_aaf(expt_dir=config['save'], 
						load_dir=config['load'],
						load_tf_dir=config['load_tf'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						attention_forcing=config['attention_forcing'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm'],
						fr_loss_max_rate=config['fr_loss_max_rate'],
						ep_aaf_start=config['ep_aaf_start']
						)

		# run training
		seq2seq_dd = t.train(train_set, seq2seq_dd, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)

	if config['train_mode'] == 'paf':
		# import pdb; pdb.set_trace()

		""" partial af """

		# make consistent
		if config['attention_forcing'] == False:
			config['attention_loss_coeff'] = 0.0

		# construct model
		seq2seq_dd = Seq2Seq_DD_paf(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=config['attention_forcing'],
								flag_stack_outputs=False).to(device)

		# contruct trainer
		t = Trainer_paf(expt_dir=config['save'], 
						load_dir=config['load'],
						load_tf_dir=config['load_tf'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						attention_forcing=config['attention_forcing'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm'],
						nb_fr_tokens_max=config['nb_fr_tokens_max']
						)

		# run training
		seq2seq_dd = t.train(train_set, seq2seq_dd, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)

	if config['train_mode'] in ['oaf', 'oaf_alwaysKL', 'oaf_noKL', 'oaf_alwaysKLsmooth', 'oaf_alwaysMSE', 'aoaf']:
		""" train either tf or af model """

		# make consistent
		if config['attention_forcing'] == False:
			config['attention_loss_coeff'] = 0.0

		# construct model
		seq2seq_dd = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=config['attention_forcing'],
								flag_stack_outputs=False).to(device)

		# contruct trainer
		def get_trainer(*args, **kwargs):
			dct_tmp = {'oaf':Trainer_oaf, 'oaf_alwaysKL':Trainer_oaf_alwaysKL, 'oaf_noKL':Trainer_oaf_noKL, 
			'oaf_alwaysKLsmooth':Trainer_oaf_alwaysKLsmooth, 'oaf_alwaysMSE': Trainer_oaf_alwaysMSE, 'aoaf': Trainer_aoaf}
			return dct_tmp[config['train_mode']](*args, **kwargs)

		# t = Trainer_oaf(expt_dir=config['save'], 
		t = get_trainer(expt_dir=config['save'], 
						load_dir=config['load'],
						load_tf_dir=config['load_tf'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						attention_forcing=config['attention_forcing'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm']
						)

		# run training
		seq2seq_dd = t.train(train_set, seq2seq_dd, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)

	elif config['train_mode'] == 'afdynamic':

		""" train af model """

		# construct model
		seq2seq_dd = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=config['attention_forcing']).to(device)

		# contruct trainer
		t = Trainer_afdynamic(expt_dir=config['save'], 
						load_dir=config['load'],
						load_tf_dir=config['load_tf'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						attention_forcing=config['attention_forcing'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm']
						)

		# run training
		seq2seq_dd = t.train(train_set, seq2seq_dd, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)

	elif config['train_mode'] == 'dual':

		""" train tf & af models simultaneously; all start from scratch """

		model_tf = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=False).to(device)

		model_af = Seq2Seq_DD(vocab_size_enc, vocab_size_dec,
								embedding_size_enc=config['embedding_size_enc'],
								embedding_size_dec=config['embedding_size_dec'],
								embedding_dropout_rate=config['embedding_dropout'],
								hidden_size_enc=config['hidden_size_enc'],
								num_bilstm_enc=config['num_bilstm_enc'],
								num_unilstm_enc=config['num_unilstm_enc'],
								hidden_size_dec=config['hidden_size_dec'],
								num_unilstm_dec=config['num_unilstm_dec'],
								hidden_size_att=config['hidden_size_att'],
								hidden_size_shared=config['hidden_size_shared'],
								dropout_rate=config['dropout'],
								residual=config['residual'],
								batch_first=config['batch_first'],
								max_seq_len=config['max_seq_len'],
								batch_size=config['batch_size'],
								load_embedding_src=config['load_embedding_src'],
								load_embedding_tgt=config['load_embedding_tgt'],
								src_word2id=train_set.src_word2id,
								tgt_word2id=train_set.tgt_word2id,
								src_id2word=train_set.src_id2word,
								tgt_id2word=train_set.tgt_id2word,
								att_mode=config['att_mode'],
								hard_att=config['hard_att'],
								use_gpu=config['use_gpu'],
								additional_key_size=config['additional_key_size'],
								attention_forcing=True).to(device)

		# contruct trainer
		t = Trainer_dual(expt_dir=config['save'], 
						load_dir=config['load'],
						batch_size=config['batch_size'],
						random_seed=config['random_seed'],
						checkpoint_every=config['checkpoint_every'],
						print_every=config['print_every'],
						learning_rate=config['learning_rate'],
						eval_with_mask=config['eval_with_mask'],
						scheduled_sampling=config['scheduled_sampling'],
						teacher_forcing_ratio=config['teacher_forcing_ratio'],
						attention_loss_coeff=config['attention_loss_coeff'],
						use_gpu=config['use_gpu'],
						max_grad_norm=config['max_grad_norm']
						)

		# run training
		model_tf, model_af = t.train(train_set, model_tf, model_af, num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)

	else:
		assert False, 'mode not implemented'

if __name__ == '__main__':
	main()






