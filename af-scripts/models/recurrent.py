import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.attention import AttentionLayer
from utils.config import PAD, EOS, BOS
from utils.misc import get_base_hidden, _inflate, inflat_hidden_state
from utils.dataset import load_pretrained_embedding

from torch.distributions import Categorical

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')

KEY_ATTN_REF = 'attention_ref'
KEY_ATTN_SCORE = 'attention_score'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'
KEY_MODEL_STRUCT = 'model_struct'

class Seq2Seq_DD(nn.Module):

	""" DD enc-dec model """

	def __init__(self, 
		# add params
		vocab_size_enc,
		vocab_size_dec,
		embedding_size_enc=200,
		embedding_size_dec=200,
		embedding_dropout_rate=0,
		hidden_size_enc=200,
		num_bilstm_enc=2,
		num_unilstm_enc=0,
		hidden_size_dec=200,
		num_unilstm_dec=2,
		hidden_size_att=10,
		hidden_size_shared=200,
		dropout_rate=0.0,
		residual=False,
		batch_first=True,
		max_seq_len=32,
		batch_size=64,
		load_embedding_src=None,
		load_embedding_tgt=None,
		src_word2id=None,
		tgt_word2id=None,
		src_id2word=None,
		tgt_id2word=None,
		att_mode='bahdanau',
		hard_att=False,
		use_gpu=False,
		additional_key_size=0,
		attention_forcing=False,
		flag_stack_outputs=False,
		flag_greedy=True
		):

		super(Seq2Seq_DD, self).__init__()
		# config device
		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')	

		# define var
		self.hidden_size_enc = hidden_size_enc
		self.num_bilstm_enc = num_bilstm_enc
		self.num_unilstm_enc= num_unilstm_enc
		self.hidden_size_dec = hidden_size_dec 
		self.num_unilstm_dec = num_unilstm_dec 
		self.hidden_size_att = hidden_size_att 
		self.hidden_size_shared = hidden_size_shared #[200] (transformed output off att)
		self.batch_size = batch_size
		self.max_seq_len = max_seq_len
		self.use_gpu = use_gpu
		self.hard_att = hard_att
		self.additional_key_size = additional_key_size
		self.residual = residual
		self.attention_forcing = attention_forcing
		self.dropout_rate = dropout_rate
		self.embedding_dropout_rate = embedding_dropout_rate

		# use separate embedding + vocab
		self.vocab_size_enc = vocab_size_enc
		self.vocab_size_dec = vocab_size_dec
		self.embedding_size_enc = embedding_size_enc
		self.embedding_size_dec = embedding_size_dec
		self.load_embedding_enc = load_embedding_src
		self.load_embedding_dec = load_embedding_tgt
		self.word2id_enc = src_word2id
		self.id2word_enc = src_id2word
		self.word2id_dec = tgt_word2id
		self.id2word_dec = tgt_id2word

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout_rate)
		self.dropout = nn.Dropout(dropout_rate)
		self.beam_width = 0 
		self.debug_count = 0

		# load embeddings
		if self.load_embedding_enc:
			embedding_matrix_enc = np.random.rand(self.vocab_size, self.embedding_size)
			embedding_matrix_enc = load_pretrained_embedding(self.word2id_enc, embedding_matrix_enc, self.load_embedding_enc)
			embedding_matrix_enc = torch.FloatTensor(embedding_matrix_enc)
			self.embedder_enc = nn.Embedding.from_pretrained(embedding_matrix_enc, 
										freeze=False, sparse=False, padding_idx=PAD)        
		else:
			self.embedder_enc = nn.Embedding(self.vocab_size_enc, self.embedding_size_enc,
										sparse=False, padding_idx=PAD)      

		if self.load_embedding_dec:
			embedding_matrix_dec = np.random.rand(self.vocab_size, self.embedding_size)
			embedding_matrix_dec = load_pretrained_embedding(self.word2id_dec, embedding_matrix_dec, self.load_embedding_dec)
			embedding_matrix_dec = torch.FloatTensor(embedding_matrix_dec)
			self.embedder_dec = nn.Embedding.from_pretrained(embedding_matrix_dec, 
										freeze=False, sparse=False, padding_idx=PAD)        
		else:
			self.embedder_dec = nn.Embedding(self.vocab_size_dec, self.embedding_size_dec,
										sparse=False, padding_idx=PAD)      
		# define enc
		# embedding_size_enc -> hidden_size_enc * 2
		self.enc = torch.nn.LSTM(self.embedding_size_enc, self.hidden_size_enc, 
								num_layers=self.num_bilstm_enc, batch_first=batch_first, 
								bias=True, dropout=dropout_rate,
								bidirectional=True)

		if self.num_unilstm_enc != 0:
			if not self.residual:
				self.enc_uni = torch.nn.LSTM(self.hidden_size_enc * 2, self.hidden_size_enc * 2, 
										num_layers=self.num_unilstm_enc, batch_first=batch_first, 
										bias=True, dropout=dropout_rate,
										bidirectional=False)
			else:
				self.enc_uni = nn.Module()
				for i in range(self.num_unilstm_enc):
					self.enc_uni.add_module(
						'l'+str(i), 
						torch.nn.LSTM(self.hidden_size_enc * 2, self.hidden_size_enc * 2, 
										num_layers=1, batch_first=batch_first, bias=True, 
										dropout=dropout_rate,bidirectional=False))

		# define dec
		# embedding_size_dec + self.hidden_size_shared [200+200] -> hidden_size_dec [200]
		if not self.residual:
			self.dec = torch.nn.LSTM(self.embedding_size_dec + self.hidden_size_shared, 
									self.hidden_size_dec, 
									num_layers=self.num_unilstm_dec, batch_first=batch_first, 
									bias=True, dropout=dropout_rate,
									bidirectional=False)
		else:
			lstm_uni_dec_first = torch.nn.LSTM(self.embedding_size_dec + self.hidden_size_shared, 
									self.hidden_size_dec, 
									num_layers=1, batch_first=batch_first, 
									bias=True, dropout=dropout_rate,
									bidirectional=False)
			self.dec = nn.Module()
			self.dec.add_module('l0', lstm_uni_dec_first)
			for i in range(1, self.num_unilstm_dec):
				self.dec.add_module(
					'l'+str(i),
					torch.nn.LSTM(self.hidden_size_dec, self.hidden_size_dec, 
									num_layers=1, batch_first=batch_first, bias=True, 
									dropout=dropout_rate, bidirectional=False))

		# define att
		# query: hidden_size_dec [200]
		# keys: hidden_size_enc * 2 + (optional) self.additional_key_size [400]
		# values: hidden_size_enc * 2 [400]
		# context: weighted sum of values [400]
		self.key_size = self.hidden_size_enc * 2 + self.additional_key_size
		self.value_size = self.hidden_size_enc * 2
		self.query_size = self.hidden_size_dec
		self.att = AttentionLayer(self.query_size, self.key_size, value_size=self.value_size,
									mode=att_mode, dropout=dropout_rate, 
									query_transform=False, output_transform=False,
									hidden_size=self.hidden_size_att, use_gpu=self.use_gpu, hard_att=self.hard_att)

		# define output
		# (hidden_size_enc * 2 + hidden_size_dec) -> self.hidden_size_shared -> vocab_size_dec
		self.ffn = nn.Linear(self.hidden_size_enc * 2 + self.hidden_size_dec , self.hidden_size_shared, bias=False)    
		self.out = nn.Linear(self.hidden_size_shared , self.vocab_size_dec, bias=True)

		self.flag_stack_outputs = flag_stack_outputs
		self.flag_greedy = flag_greedy


	def reset_dropout(self, dropout_rate):

		self.dropout = nn.Dropout(dropout_rate)

		# define enc
		# embedding_size_enc -> hidden_size_enc * 2
		self.enc.dropout = dropout_rate
		if self.num_unilstm_enc != 0:
			if not self.residual:
				self.enc_uni.dropout = dropout_rate
			else:
				for i in range(self.num_unilstm_enc):
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					enc_func.dropout = dropout_rate

		# define dec
		# embedding_size_dec + self.hidden_size_shared [200+200] -> hidden_size_dec [200]
		if not self.residual:
			self.dec.dropout = dropout_rate
		else:
			for i in range(0, self.num_unilstm_dec):
				dec_func = getattr(self.dec, 'l'+str(i))
				dec_func.dropout = dropout_rate

		# define att
		self.att.dropout = torch.nn.Dropout(dropout_rate)


	def reset_use_gpu(self, use_gpu):

		self.use_gpu = use_gpu

	def reset_max_seq_len(self, max_seq_len):

		self.max_seq_len = max_seq_len

	def reset_batch_size(self, batch_size):

		self.batch_size = batch_size

	def set_beam_width(self, beam_width):

		self.beam_width = beam_width

	def set_var(self, var_name, var_val):

		""" set variable value """

		setattr(self, var_name, var_val)


	def check_classvar(self, var_name):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			if var_name == 'additional_key_size':
				var_val = 0
			elif var_name == 'attention_forcing':
				var_val = False
			elif var_name == 'num_unilstm_enc':
				var_val = 0
			elif var_name == 'residual':
				var_val = False
			elif var_name == 'flag_greedy':
				var_val = True
			else:
				var_val = None

			# set class attribute to default value
			setattr(self, var_name, var_val)


	def forward(self, src, tgt=None, 
		hidden=None, is_training=False, teacher_forcing_ratio=1.0,
		att_key_feats=None, att_scores=None, beam_width=0):

		"""
			Args:
				src: list of src word_ids [batch_size, max_seq_len, word_ids]
				tgt: list of tgt word_ids
				hidden: initial hidden state
				is_training: whether in eval or train mode
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output - log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				ret_dict
		"""

		# import pdb; pdb.set_trace()

		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')	
			
		# ******************************************************
		# 0. init var
		ret_dict = dict()
		ret_dict[KEY_ATTN_SCORE] = []
		ret_dict[KEY_ATTN_REF] = []

		decoder_outputs = []
		sequence_symbols = []
		batch_size = self.batch_size
		lengths = np.array([self.max_seq_len] * batch_size)
		self.beam_width = beam_width

		# src mask
		mask_src = src.data.eq(PAD)
		# print(mask_src[0])

		# ******************************************************
		# 1. convert id to embedding 
		emb_src = self.embedding_dropout(self.embedder_enc(src))
		if type(tgt) == type(None):
			tgt = torch.Tensor([BOS]).repeat(src.size()).type(torch.LongTensor).to(device=device)
		emb_tgt = self.embedding_dropout(self.embedder_dec(tgt))

		# ******************************************************
		# 2. run enc 
		enc_outputs, enc_hidden = self.enc(emb_src, hidden)
		enc_outputs = self.dropout(enc_outputs)\
						.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				enc_hidden_uni_init = None
				enc_outputs, enc_hidden_uni = self.enc_uni(enc_outputs, enc_hidden_uni_init)
				enc_outputs = self.dropout(enc_outputs)\
								.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))
			else:
				enc_hidden_uni_init = None
				enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					enc_inputs = enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					enc_outputs, enc_hidden_uni = enc_func(enc_inputs, enc_hidden_uni_init)
					enc_hidden_uni_lis.append(enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						enc_outputs = enc_outputs + enc_inputs
					enc_outputs = self.dropout(enc_outputs)\
									.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))

		# ******************************************************
		# 2.5 att inputs: keys n values 
		if type(att_key_feats) == type(None):
			att_keys = enc_outputs
		else:
			# att_key_feats: b x max_seq_len x additional_key_size
			assert self.additional_key_size == att_key_feats.size(-1), 'Mismatch in attention key dimension!'
			att_keys = torch.cat((enc_outputs, att_key_feats), dim=2)
		att_vals = enc_outputs
		# print(att_keys.size())

		# ******************************************************
		# 3. init hidden states - TODO 
		dec_hidden = None

		# ======================================================
		# decoder
		def decode(step, step_output, step_attn):
			
			"""
				Greedy decoding
				Note:
					it should generate EOS, PAD as used in training tgt
				Args:
					step: step idx
					step_output: log predicted_softmax [batch_size, 1, vocab_size_dec]
					step_attn: attention scores - (batch_size x tgt_len(query_len) x src_len(key_len)
				Returns:
					symbols: most probable symbol_id [batch_size, 1] if flag_greedy==True
			"""

			ret_dict[KEY_ATTN_SCORE].append(step_attn)
			decoder_outputs.append(step_output)
			if self.flag_greedy:
				symbols = decoder_outputs[-1].topk(1)[1]
			else:
				symbols = Categorical(logits=decoder_outputs[-1]).sample().unsqueeze(1)
			sequence_symbols.append(symbols)

			# print(decoder_outputs[-1].size(), decoder_outputs[-1][0])
			# print(symbols.size(), symbols[0])
			# import pdb; pdb.set_trace()
			# print(symbols)
			# input('...')

			eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD)) # equivalent to logical OR
			# eos_batches = symbols.data.eq(PAD) 
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
				# print(lengths)
				# input('...')
			return symbols
		# ======================================================


		# ******************************************************
		# 4. run dec + att + shared + output
		"""
			teacher_forcing_ratio = 1.0 -> always teacher forcing

			E.g.: 
				emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
				tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
				predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
				(shift-by-1)
		"""
		attention_forcing = self.attention_forcing
		if not is_training:
			 attention_forcing = False
			 teacher_forcing_ratio = 0.0

		# beam search decoding
		if not is_training and self.beam_width > 1:
			decoder_outputs, decoder_hidden, metadata = \
					self.beam_search_decoding(att_keys, att_vals,
					dec_hidden, mask_src, beam_width=self.beam_width)

			return decoder_outputs, decoder_hidden, metadata

		# no beam search decoding 
		tgt_chunk = emb_tgt[:, 0].unsqueeze(1) # BOS
		cell_value = torch.FloatTensor([0]).repeat(self.batch_size, 1, self.hidden_size_shared).to(device=device)
		prev_c = torch.FloatTensor([0]).repeat(self.batch_size, 1, self.max_seq_len).to(device=device)

		if not attention_forcing:

			# tf at sentence level
			use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
			for idx in range(self.max_seq_len - 1):

				predicted_softmax, dec_hidden, step_attn, c_out, cell_value = \
					self.forward_step(att_keys, att_vals, tgt_chunk, cell_value,
										dec_hidden, mask_src, prev_c)
				step_output = predicted_softmax.squeeze(1)
				symbols = decode(idx, step_output, step_attn)
				# print(symbols)
				# print(tgt[0][idx+1])
				prev_c = c_out
				if use_teacher_forcing:
					tgt_chunk = emb_tgt[:, idx+1].unsqueeze(1)
					if self.debug_count < 1:
						print('w/o attention forcing + w/ teacher forcing')
						self.debug_count += 1
					# print('here')
					# print(tgt[:, idx+1])
					# print(symbols.view(-1))
					# input('...')
				else:
					tgt_chunk = self.embedder_dec(symbols)
					if self.debug_count < 1:
						print('w/o attention forcing + w/o teacher forcing')
						self.debug_count += 1
				# print('target query size: {}'.format(tgt_chunk.size()))
				# print(lengths)
		else:
			# init
			# print('here')
			tgt_chunk_ref = tgt_chunk
			tgt_chunk_hyp = tgt_chunk
			cell_value_ref = cell_value
			cell_value_hyp = cell_value
			prev_c_ref = prev_c
			prev_c_hyp = prev_c
			dec_hidden_ref = dec_hidden
			dec_hidden_hyp = dec_hidden

			# tf at sentence level
			use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
			# loop
			for idx in range(self.max_seq_len - 1):

				assert type(att_scores) != type(None), 'empty att ref scores!'
				if isinstance(att_scores, list):
					# used in dual training
					step_attn_ref_detach = att_scores[idx].detach() 
				else:
					# used in fixed ref att training
					step_attn_ref_detach = att_scores[:, idx,:].unsqueeze(1)
				step_attn_ref_detach = step_attn_ref_detach.type(torch.FloatTensor).to(device=device)
				# print(step_attn_ref_detach.size())
				# print('here')
				# input('...')
				if self.debug_count < 1:
					print('w/ attention forcing')

				# hyp
				ret_dict[KEY_ATTN_REF].append(step_attn_ref_detach)
				predicted_softmax_hyp, dec_hidden_hyp, step_attn_hyp, c_out_hyp, cell_value_hyp = \
					self.forward_step(att_keys, att_vals, tgt_chunk_hyp, cell_value_hyp,
										dec_hidden_hyp, mask_src, prev_c_hyp, att_ref=step_attn_ref_detach)
				step_output_hyp = predicted_softmax_hyp.squeeze(1)
				symbols_hyp = decode(idx, step_output_hyp, step_attn_hyp)
				prev_c_hyp = c_out_hyp

				if use_teacher_forcing:
					tgt_chunk_hyp = emb_tgt[:, idx+1].unsqueeze(1)
					# print('here')
					# print(tgt_chunk)
					if self.debug_count < 1:
						print('w/ teacher forcing')
						self.debug_count += 1

				else:
					tgt_chunk_hyp = self.embedder_dec(symbols_hyp)
					if self.debug_count < 1:
						print('w/o teacher forcing')
						self.debug_count += 1

				# tgt_chunk_hyp = self.embedder_dec(symbols_hyp)
				# print('step_attn_hyp', step_attn_hyp)
				# print(step_attn_ref_detach)
				# print(step_attn_hyp)
				# print(src)
				# print(tgt)
				# input('...')

		# print('...')
		ret_dict[KEY_SEQUENCE] = sequence_symbols
		ret_dict[KEY_LENGTH] = lengths.tolist()

		if hasattr(self, 'flag_stack_outputs'):
			if self.flag_stack_outputs:
				# print('stack')
				decoder_outputs = torch.stack(decoder_outputs, dim=2)
				ret_dict['attention_score'] = torch.cat(ret_dict['attention_score'], dim=1)
				ret_dict['attention_ref'] = torch.cat(ret_dict['attention_ref'], dim=1)

		return decoder_outputs, dec_hidden, ret_dict
		

	def forward_step(self, att_keys, att_vals, tgt_chunk, prev_cell_value,
		dec_hidden=None, mask_src=None, prev_c=None, att_ref=None, use_gpu=True):

		"""
			manual unrolling - can only operate per time step 

			Args:
				att_keys:   [batch_size, seq_len, hidden_size_enc * 2 + optional key size (key_size)]
				att_vals:   [batch_size, seq_len, hidden_size_enc * 2 (val_size)]
				tgt_chunk:  tgt word embeddings 
							non teacher forcing - [batch_size, 1, embedding_size_dec] (lose 1 dim when indexed)
				prev_cell_value:
							previous cell value before prediction [batch_size, 1, self.state_size]            
				dec_hidden: 
							initial hidden state for dec layer  
				mask_src: 
							mask of PAD for src sequences   
				prev_c:     
							used in hybrid attention mechanism  
				att_ref:
							reference attention scores (used to calculate weighted sum)
	
			Returns:    
				predicted_softmax: log probilities [batch_size, vocab_size_dec]
				dec_hidden: a list of hidden states of each dec layer
				attn: attention weights
				cell_value: transformed attention output [batch_size, 1, self.hidden_size_shared]
		"""

		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		# record sizes
		batch_size = tgt_chunk.size(0)
		tgt_chunk = torch.cat([tgt_chunk, prev_cell_value], -1)
		tgt_chunk = tgt_chunk.view(-1, 1, self.embedding_size_dec + self.hidden_size_shared)

		# run dec
		# print('-- run dec --')    
		# default dec_hidden: [h_0, c_0]; with h_0 [num_layers * num_directions(==1), batch, hidden_size]
		if not self.residual:
			dec_outputs, dec_hidden = self.dec(tgt_chunk, dec_hidden)
			dec_outputs = self.dropout(dec_outputs)
		else:
			# store states layer by layer num_layers * ([1, batch, hidden_size], [1, batch, hidden_size])
			dec_hidden_lis = [] 

			# layer0
			dec_func_first = getattr(self.dec, 'l0')
			if type(dec_hidden) == type(None):
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk, None)
			else:
				index = torch.tensor([0]).to(device=device) # choose the 0th layer
				dec_hidden_in = tuple([h.index_select(dim=0, index=index) for h in dec_hidden])
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk, dec_hidden_in)
			dec_hidden_lis.append(dec_hidden_out)
			# print(type(dec_hidden_out[0]))
			# input('...')
			# no residual for 0th layer
			dec_outputs = self.dropout(dec_outputs)

			# layer1+
			for i in range(1, self.num_unilstm_dec):
				dec_inputs = dec_outputs
				dec_func = getattr(self.dec, 'l'+str(i))
				if type(dec_hidden) == type(None):
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, None)
				else:
					index = torch.tensor([i]).to(device=device) 
					dec_hidden_in = tuple([h.index_select(dim=0, index=index) for h in dec_hidden])					
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, dec_hidden_in)
				dec_hidden_lis.append(dec_hidden_out)
				if i < self.num_unilstm_dec - 1:
					dec_outputs = dec_outputs + dec_inputs
				dec_outputs = self.dropout(dec_outputs)
			
			# convert to tuple
			h_0 = torch.cat([h[0] for h in dec_hidden_lis], 0)
			c_0 = torch.cat([h[1] for h in dec_hidden_lis], 0)
			dec_hidden = tuple([h_0, c_0])
			# print(dec_hidden[0])
			# print(dec_hidden[0].size())
			# input('...')

		# run att 
		# print('-- run att --')        
		self.att.set_mask(mask_src) 
		att_outputs, attn, c_out = self.att(dec_outputs, att_keys, att_vals, prev_c=prev_c, att_ref=att_ref)
		att_outputs = self.dropout(att_outputs)
		# print('att output size: {}'.format(att_outputs.size()))

		# run ff + softmax
		# print('-- run output --')     
		ff_inputs = torch.cat((att_outputs, dec_outputs), dim=-1)
		ff_inputs_size = self.hidden_size_enc * 2 + self.hidden_size_dec
		cell_value = self.ffn(ff_inputs.view(-1, 1, ff_inputs_size)) # 600 -> 200
		outputs = self.out(cell_value.contiguous().view(-1, self.hidden_size_shared))
		predicted_softmax = F.log_softmax(outputs, dim=1).view(batch_size, 1, -1)
		# print('outputs size: {}'.format(predicted_softmax.size()))

		# print('-----------------------------')    
		return predicted_softmax, dec_hidden, attn, c_out, cell_value


	def beam_search_decoding(self, att_keys, att_vals,
		dec_hidden=None, 
		mask_src=None, prev_c=None, beam_width=10):

		"""
			beam search decoding - only used for evaluation
			Modified from - https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py
			
			Shortcuts:
				beam_width: k
				batch_size: b
				vocab_size: v
				max_seq_len: l

			Args:
				att_keys:   [b x l x hidden_size_enc * 2 + optional key size (key_size)]
				att_vals:   [b x l x hidden_size_enc * 2 (val_size)]
				dec_hidden: 
							initial hidden state for dec layer [b x h_dec]
				mask_src: 
							mask of PAD for src sequences   
				beam_width: beam width kept during searching
	
			Returns:    
				decoder_outputs: output probabilities [(batch, 1, vocab_size)] * T
				decoder_hidden (num_layers * num_directions, batch, hidden_size): 
										tensor containing the last hidden state of the decoder.
				ret_dict: dictionary containing additional information as follows 
						{	
							*length* : list of integers representing lengths of output sequences, 
							*topk_length*: list of integers representing lengths of beam search sequences, 
							*sequence* : list of sequences, where each sequence is a list of predicted token IDs,
							*topk_sequence* : list of beam search sequences, each beam is a list of token IDs, 
							*outputs* : [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
						}.
		"""

		# define var
		self.beam_width = beam_width
		self.pos_index = Variable(torch.LongTensor(range(self.batch_size)) * self.beam_width).view(-1, 1).to(device=device)
		# print('---pos_index---')
		# print(self.pos_index)
		# input('...')

		# initialize the input vector; att_c_value
		input_var = Variable(torch.transpose(torch.LongTensor([[BOS] * self.batch_size * self.beam_width]), 0, 1)).to(device=device)
		input_var_emb = self.embedder_dec(input_var).to(device=device)
		prev_c = torch.FloatTensor([0]).repeat(self.batch_size, 1, self.max_seq_len).to(device=device)
		cell_value = torch.FloatTensor([0]).repeat(self.batch_size, 1, self.hidden_size_shared).to(device=device)		
		# print('---init input BOS---')
		# print(input_var)
		# input('...')

		# inflate attention keys and values (derived from encoder outputs)
		# wrong ordering
		# inflated_att_keys = _inflate(att_keys, self.beam_width, 0)
		# inflated_att_vals = _inflate(att_vals, self.beam_width, 0)
		# inflated_mask_src = _inflate(mask_src, self.beam_width, 0)
		# inflated_prev_c = _inflate(prev_c, self.beam_width, 0)
		# correct ordering
		inflated_att_keys = att_keys.repeat_interleave(self.beam_width, dim=0)
		inflated_att_vals = att_vals.repeat_interleave(self.beam_width, dim=0)
		inflated_mask_src = mask_src.repeat_interleave(self.beam_width, dim=0)
		inflated_prev_c = prev_c.repeat_interleave(self.beam_width, dim=0)
		inflated_cell_value = cell_value.repeat_interleave(self.beam_width, dim=0)
		# print('---inflated vars---')
		# print('inflated_att_keys')
		# print(inflated_att_keys)
		# print(inflated_att_keys.size())
		# print('inflated_att_vals')
		# print(inflated_att_vals)
		# print(inflated_att_vals.size())
		# print('inflated_mask_src')
		# print(inflated_mask_src)
		# print(inflated_mask_src.size())
		# print('inflated_prev_c')
		# print(inflated_prev_c)
		# print(inflated_prev_c.size())
		# input('...')

		# inflate hidden states and others
		# note that inflat_hidden_state might be faulty - currently using None so it's fine
		dec_hidden = inflat_hidden_state(dec_hidden, self.beam_width)
		# print('---inflated hidden states---')
		# print('dec_hidden')
		# print(dec_hidden)
		# input('...')

		# Initialize the scores; for the first step,
		# ignore the inflated copies to avoid duplicate entries in the top k
		sequence_scores = torch.Tensor(self.batch_size * self.beam_width, 1).to(device=device)
		sequence_scores.fill_(-float('Inf'))
		sequence_scores.index_fill_(0, torch.LongTensor([i * self.beam_width for i in range(0, self.batch_size)]).to(device=device), 0.0)
		sequence_scores = Variable(sequence_scores)
		# print('---sequence_scores---')
		# print(sequence_scores)
		# print(sequence_scores.size())
		# input('...')

		# Store decisions for backtracking
		stored_outputs = list()         # raw softmax scores [bk x v] * T
		stored_scores = list()          # topk scores [bk] * T
		stored_predecessors = list()    # preceding beam idx (from 0-bk) [bk] * T
		stored_emitted_symbols = list() # word ids [bk] * T
		stored_hidden = list()          # 

		for _ in range(self.max_seq_len):

			predicted_softmax, dec_hidden, step_attn, inflated_c_out, inflated_cell_value = \
				self.forward_step(inflated_att_keys, inflated_att_vals, input_var_emb, inflated_cell_value, 
								dec_hidden, inflated_mask_src, inflated_prev_c)
			inflated_prev_c = inflated_c_out
			# print('---step results---')
			# print('predicted_softmax')
			# print(predicted_softmax)
			# print(predicted_softmax.size())
			# print('dec_hidden')
			# print(dec_hidden)
			# print('step_attn')
			# print(step_attn)
			# print(step_attn.size())		
			# print('inflated_prev_c')
			# print(inflated_prev_c)
			# # print(inflated_prev_c.size())
			# input('...')

			# retain output probs
			stored_outputs.append(predicted_softmax) # [bk x v]

			# To get the full sequence scores for the new candidates, 
			# add the local scores for t_i to the predecessor scores for t_(i-1)
			sequence_scores = _inflate(sequence_scores, self.vocab_size_dec, 1)
			sequence_scores += predicted_softmax.squeeze(1) # [bk x v]
			# print('---scores---')
			# print('predicted_softmax', predicted_softmax.squeeze(1))
			# print('sequence_scores', sequence_scores)
			# print(sequence_scores.size())
			# print('sequence_scores squeeze', sequence_scores.view(self.batch_size, -1))
			# print(sequence_scores.view(self.batch_size, -1).size())

			scores, candidates = sequence_scores.view(self.batch_size, -1).topk(self.beam_width, dim=1) # [b x kv] -> [b x k]
			# print('scores', scores)
			# print(scores.size())
			# print('candidates', candidates)
			# print(candidates.size())
			# input('...')

			# Reshape input = (bk, 1) and sequence_scores = (bk, 1)
			input_var = (candidates % self.vocab_size_dec).view(self.batch_size * self.beam_width, 1).to(device=device)
			input_var_emb = self.embedder_dec(input_var)
			sequence_scores = scores.view(self.batch_size * self.beam_width, 1) #[bk x 1]
			# print('input_var', input_var)
			# print(input_var.size())
			# print(input_var_emb.size())
			# print('sequence_scores', sequence_scores)
			# print(sequence_scores.size())

			# Update fields for next timestep

			predecessors = (candidates / self.vocab_size_dec + self.pos_index.expand_as(candidates)).\
							view(self.batch_size * self.beam_width, 1)
			# print('---predecessor---')
			# print('candidates', candidates)
			# print('predecessors', predecessors)
			# input('...')

			# dec_hidden: [h_0, c_0]; with h_0 [num_layers * num_directions, batch, hidden_size]
			if isinstance(dec_hidden, tuple):
				# print('---dec_hidden---')
				# print('predecessors squeeze', predecessors.squeeze())
				# print('dec_hidden h0 before', dec_hidden[0])
				dec_hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in dec_hidden])
				# print('dec_hidden h0 after', dec_hidden[0])
				# print(dec_hidden[0].size())
				# input('...')
			else:
				dec_hidden = dec_hidden.index_select(1, predecessors.squeeze())             

			# Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
			# eos_indices = input_var.data.eq(EOS)
			# pad_indices = input_var.data.eq(PAD)
			# if eos_indices.nonzero().dim() > 0:
			# 	sequence_scores.data.masked_fill_(eos_indices, -float('inf'))
			# if pad_indices.nonzero().dim() > 0:
			# 	sequence_scores.data.masked_fill_(pad_indices, -float('inf'))

			stored_scores.append(sequence_scores.clone())
			# print('eos_indices', eos_indices)
			# print('pad_indices', pad_indices)
			# print('sequence_scores', sequence_scores)
			# input('...')

			# Cache results for backtracking
			stored_predecessors.append(predecessors)
			stored_emitted_symbols.append(input_var)
			stored_hidden.append(dec_hidden)

		# print('---stored scores---')
		# print(stored_scores)
		# print(stored_emitted_symbols)
		# input('...')
		# Do backtracking to return the optimal values
		output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
													stored_predecessors, stored_emitted_symbols,
													stored_scores, self.batch_size, self.hidden_size_dec)

		# Build return objects
		decoder_outputs = [step[:, 0, :].squeeze(1) for step in output]
		if isinstance(h_n, tuple):
			decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
		else:
			decoder_hidden = h_n[:, :, 0, :]
		metadata = {}
		metadata['output'] = output
		metadata['h_t'] = h_t
		metadata['score'] = s
		metadata['topk_length'] = l
		metadata['topk_sequence'] = p # [b x k x 1] * T
		metadata['length'] = [seq_len[0] for seq_len in l]
		metadata['sequence'] = [seq[:, 0] for seq in p]

		return decoder_outputs, decoder_hidden, metadata


	def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):

		"""
			Backtracks over batch to generate optimal k-sequences.
			https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py

			Args:
				nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
				nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
				predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
				symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
				scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
				b: Size of the batch
				hidden_size: Size of the hidden state

			Returns:
				output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
				from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
				h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
				from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
				h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
				score [batch, k]: A list containing the final scores for all top-k sequences
				length [batch, k]: A list specifying the length of each sequence in the top-k candidates
				p (batch, k, sequence_len): A Tensor containing predicted sequence [b x k x 1] * T
		"""

		# initialize return variables given different types
		output = list()
		h_t = list()
		p = list()

		# Placeholder for last hidden state of top-k sequences.
		# If a (top-k) sequence ends early in decoding, `h_n` contains
		# its hidden state when it sees EOS.  Otherwise, `h_n` contains
		# the last hidden state of decoding.
		lstm = isinstance(nw_hidden[0], tuple)
		if lstm:
			state_size = nw_hidden[0][0].size()
			h_n = tuple([torch.zeros(state_size).to(device=device), torch.zeros(state_size).to(device=device)])
		else:
			h_n = torch.zeros(nw_hidden[0].size()).to(device=device)
		# print('---h_n---')
		# print(h_n)
		# print(h_n[0].size())
		# input('...')

		# Placeholder for lengths of top-k sequences
		# Similar to `h_n`
		l = [[self.max_seq_len] * self.beam_width for _ in range(b)]  
		# print('---length---')
		# print(l)
		# input('...')		

		# the last step output of the beams are not sorted
		# thus they are sorted here
		sorted_score, sorted_idx = scores[-1].view(b, self.beam_width).topk(self.beam_width)
		sorted_score = sorted_score.to(device=device)
		sorted_idx = sorted_idx.to(device=device)
		# print('---sorted vals---')
		# print('sorted_score', sorted_score)
		# print('sorted_idx', sorted_idx)
		# input('...')	

		# initialize the sequence scores with the sorted last step beam scores
		s = sorted_score.clone().to(device=device)

		batch_eos_found = [0] * b   # the number of EOS found
									# in the backward loop below for each batch

		t = self.max_seq_len - 1
		# initialize the back pointer with the sorted order of the last step beams.
		# add self.pos_index for indexing variable with b*k as the first dimension.
		t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.beam_width).to(device=device)
		# print('t_predecessors', t_predecessors)
		# input('...')	


		while t >= 0:
			# Re-order the variables with the back pointer
			current_output = nw_output[t].index_select(0, t_predecessors)
			if lstm:
				current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
			else:
				current_hidden = nw_hidden[t].index_select(1, t_predecessors)
			current_symbol = symbols[t].index_select(0, t_predecessors)
			# print('---loop {}---'.format(t))
			# print('t_predecessors', t_predecessors)
			# print('current_output', current_output)
			# print(current_output.size())
			# print('current_hidden', current_hidden)
			# print(current_hidden[0].size())
			# print('current_symbol', current_symbol)
			# print(current_symbol.size())
			# input('...')

			# Re-order the back pointer of the previous step with the back pointer of
			# the current step
			t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze().to(device=device)

			"""
				This tricky block handles dropped sequences that see EOS earlier.
				The basic idea is summarized below:
				
				  Terms:
					  Ended sequences = sequences that see EOS early and dropped
					  Survived sequences = sequences in the last step of the beams
				
					  Although the ended sequences are dropped during decoding,
				  their generated symbols and complete backtracking information are still
				  in the backtracking variables.
				  For each batch, everytime we see an EOS in the backtracking process,
					  1. If there is survived sequences in the return variables, replace
					  the one with the lowest survived sequence score with the new ended
					  sequences
					  2. Otherwise, replace the ended sequence with the lowest sequence
					  score with the new ended sequence
			"""

			eos_indices = symbols[t].data.squeeze(1).eq(EOS).nonzero().to(device=device)
			if eos_indices.dim() > 0:
				for i in range(eos_indices.size(0)-1, -1, -1):
					# Indices of the EOS symbol for both variables
					# with b*k as the first dimension, and b, k for
					# the first two dimensions
					idx = eos_indices[i]
					b_idx = int(idx[0] / self.beam_width)
					# The indices of the replacing position
					# according to the replacement strategy noted above
					res_k_idx = self.beam_width - (batch_eos_found[b_idx] % self.beam_width) - 1
					batch_eos_found[b_idx] += 1
					res_idx = b_idx * self.beam_width + res_k_idx

					# Replace the old information in return variables
					# with the new ended sequence information
					t_predecessors[res_idx] = predecessors[t][idx[0]].to(device=device)
					current_output[res_idx, :] = nw_output[t][idx[0], :].to(device=device)
					if lstm:
						current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].to(device=device)
						current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].to(device=device)
						h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data.to(device=device)
						h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data.to(device=device)
					else:
						current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :].to(device=device)
						h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data.to(device=device)
					current_symbol[res_idx, :] = symbols[t][idx[0]].to(device=device)
					s[b_idx, res_k_idx] = scores[t][idx[0]].data[0].to(device=device)
					l[b_idx][res_k_idx] = t + 1

			# record the back tracked results
			output.append(current_output)
			h_t.append(current_hidden)
			p.append(current_symbol)

			t -= 1

		# Sort and re-order again as the added ended sequences may change
		# the order (very unlikely)
		s, re_sorted_idx = s.topk(self.beam_width)
		for b_idx in range(b):
			l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

		re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.beam_width).to(device=device)

		# Reverse the sequences and re-order at the same time
		# It is reversed because the backtracking happens in reverse time order
		output = [step.index_select(0, re_sorted_idx).view(b, self.beam_width, -1) for step in reversed(output)]
		p = [step.index_select(0, re_sorted_idx).view(b, self.beam_width, -1) for step in reversed(p)]
		if lstm:
			h_t = [tuple([h.index_select(1, re_sorted_idx.to(device=device)).view(-1, b, self.beam_width, hidden_size) for h in step]) for step in reversed(h_t)]
			h_n = tuple([h.index_select(1, re_sorted_idx.data.to(device=device)).view(-1, b, self.beam_width, hidden_size) for h in h_n])
		else:
			h_t = [step.index_select(1, re_sorted_idx.to(device=device)).view(-1, b, self.beam_width, hidden_size) for step in reversed(h_t)]
			h_n = h_n.index_select(1, re_sorted_idx.data.to(device=device)).view(-1, b, self.beam_width, hidden_size)
		s = s.data

		return output, h_t, h_n, s, l, p


	# ===================================================================================
	# ===================================================================================
	# split the forward function into multiple functions; 
	# so that they can be called externally
	# ===================================================================================
	# ===================================================================================

	def forward_prep_attkeys(self, src, tgt, hidden=None):

		# ******************************************************
		# 0. init
		batch_size = self.batch_size # allow change in batch_size for resume
		lengths = np.array([self.max_seq_len] * batch_size)

		# ******************************************************
		# 1. convert id to embedding 
		emb_src = self.embedding_dropout(self.embedder_enc(src))
		if type(tgt) == type(None):
			tgt = torch.Tensor([BOS]).repeat(src.size()).type(torch.LongTensor).to(device=device)
		emb_tgt = self.embedding_dropout(self.embedder_dec(tgt))
		# import pdb; pdb.set_trace()

		# ******************************************************
		# 2. run enc 
		enc_outputs, enc_hidden = self.enc(emb_src, hidden)
		enc_outputs = self.dropout(enc_outputs)\
						.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				enc_hidden_uni_init = None
				enc_outputs, enc_hidden_uni = self.enc_uni(enc_outputs, enc_hidden_uni_init)
				enc_outputs = self.dropout(enc_outputs)\
								.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))
			else:
				enc_hidden_uni_init = None
				enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					enc_inputs = enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					enc_outputs, enc_hidden_uni = enc_func(enc_inputs, enc_hidden_uni_init)
					enc_hidden_uni_lis.append(enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						enc_outputs = enc_outputs + enc_inputs
					enc_outputs = self.dropout(enc_outputs)\
									.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))

		# ******************************************************
		# 2.5 att inputs: keys n values 
		att_keys = enc_outputs
		att_vals = enc_outputs
		# import pdb; pdb.set_trace()
		
		return emb_src, emb_tgt, att_keys, att_vals


	def forward_decode(self, step, step_output, lengths, sequence_symbols):

		symbols = step_output.topk(1)[1]
		sequence_symbols.append(symbols)
		eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD)) # equivalent to logical OR
		# eos_batches = symbols.data.eq(PAD) 
		if eos_batches.dim() > 0:
			eos_batches = eos_batches.cpu().view(-1).numpy()
			update_idx = ((lengths > step) & eos_batches) != 0
			lengths[update_idx] = len(sequence_symbols)
			# print(lengths)
			# input('...')
		return symbols, lengths, sequence_symbols



class Seq2Seq_DD_paf(Seq2Seq_DD):
	"""
	same model;
	the difference is AF -> partial AF, hence the arg nb_tf_tokens;
	the split is for compatibility
	"""
	def forward(self, src, tgt=None, 
		hidden=None, is_training=False, teacher_forcing_ratio=1.0,
		att_key_feats=None, att_scores=None, beam_width=0, nb_tf_tokens=0):

		"""
			Args:
				src: list of src word_ids [batch_size, max_seq_len, word_ids]
				tgt: list of tgt word_ids
				hidden: initial hidden state
				is_training: whether in eval or train mode
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output - log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				ret_dict
		"""

		# import pdb; pdb.set_trace()

		if self.use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')	
			
		# ******************************************************
		# 0. init var
		ret_dict = dict()
		ret_dict[KEY_ATTN_SCORE] = []
		ret_dict[KEY_ATTN_REF] = []

		decoder_outputs = []
		sequence_symbols = []
		batch_size = self.batch_size
		lengths = np.array([self.max_seq_len] * batch_size)
		self.beam_width = beam_width

		# src mask
		mask_src = src.data.eq(PAD)
		# print(mask_src[0])

		# ******************************************************
		# 1. convert id to embedding 
		emb_src = self.embedding_dropout(self.embedder_enc(src))
		if type(tgt) == type(None):
			tgt = torch.Tensor([BOS]).repeat(src.size()).type(torch.LongTensor).to(device=device)
		emb_tgt = self.embedding_dropout(self.embedder_dec(tgt))

		# ******************************************************
		# 2. run enc 
		enc_outputs, enc_hidden = self.enc(emb_src, hidden)
		enc_outputs = self.dropout(enc_outputs)\
						.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				enc_hidden_uni_init = None
				enc_outputs, enc_hidden_uni = self.enc_uni(enc_outputs, enc_hidden_uni_init)
				enc_outputs = self.dropout(enc_outputs)\
								.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))
			else:
				enc_hidden_uni_init = None
				enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					enc_inputs = enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					enc_outputs, enc_hidden_uni = enc_func(enc_inputs, enc_hidden_uni_init)
					enc_hidden_uni_lis.append(enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						enc_outputs = enc_outputs + enc_inputs
					enc_outputs = self.dropout(enc_outputs)\
									.view(self.batch_size, self.max_seq_len, enc_outputs.size(-1))

		# ******************************************************
		# 2.5 att inputs: keys n values 
		if type(att_key_feats) == type(None):
			att_keys = enc_outputs
		else:
			# att_key_feats: b x max_seq_len x additional_key_size
			assert self.additional_key_size == att_key_feats.size(-1), 'Mismatch in attention key dimension!'
			att_keys = torch.cat((enc_outputs, att_key_feats), dim=2)
		att_vals = enc_outputs
		# print(att_keys.size())

		# ******************************************************
		# 3. init hidden states - TODO 
		dec_hidden = None

		# ======================================================
		# decoder
		def decode(step, step_output, step_attn):
			
			"""
				Greedy decoding
				Note:
					it should generate EOS, PAD as used in training tgt
				Args:
					step: step idx
					step_output: log predicted_softmax [batch_size, 1, vocab_size_dec]
					step_attn: attention scores - (batch_size x tgt_len(query_len) x src_len(key_len)
				Returns:
					symbols: most probable symbol_id [batch_size, 1]
			"""

			ret_dict[KEY_ATTN_SCORE].append(step_attn)
			decoder_outputs.append(step_output)
			symbols = decoder_outputs[-1].topk(1)[1]
			sequence_symbols.append(symbols)
			# print(symbols)
			# input('...')

			eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD)) # equivalent to logical OR
			# eos_batches = symbols.data.eq(PAD) 
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
				# print(lengths)
				# input('...')
			return symbols
		# ======================================================


		# ******************************************************
		# 4. run dec + att + shared + output
		"""
			teacher_forcing_ratio = 1.0 -> always teacher forcing

			E.g.: 
				emb_tgt         = <s> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
				tgt_chunk in    = <s> w1 w2 w3 </s> <pad> <pad>         [max_seq_len - 1]
				predicted       =     w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len - 1]
				(shift-by-1)
		"""
		attention_forcing = self.attention_forcing
		if not is_training:
			 attention_forcing = False
			 teacher_forcing_ratio = 0.0

		# beam search decoding
		if not is_training and self.beam_width > 1:
			decoder_outputs, decoder_hidden, metadata = \
					self.beam_search_decoding(att_keys, att_vals,
					dec_hidden, mask_src, beam_width=self.beam_width)

			return decoder_outputs, decoder_hidden, metadata

		# no beam search decoding 
		tgt_chunk = emb_tgt[:, 0].unsqueeze(1) # BOS
		cell_value = torch.FloatTensor([0]).repeat(self.batch_size, 1, self.hidden_size_shared).to(device=device)
		prev_c = torch.FloatTensor([0]).repeat(self.batch_size, 1, self.max_seq_len).to(device=device)

		if not attention_forcing:

			# tf at sentence level
			use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
			for idx in range(self.max_seq_len - 1):

				predicted_softmax, dec_hidden, step_attn, c_out, cell_value = \
					self.forward_step(att_keys, att_vals, tgt_chunk, cell_value,
										dec_hidden, mask_src, prev_c)
				step_output = predicted_softmax.squeeze(1)
				symbols = decode(idx, step_output, step_attn)
				# print(symbols)
				# print(tgt[0][idx+1])
				prev_c = c_out
				if use_teacher_forcing:
					tgt_chunk = emb_tgt[:, idx+1].unsqueeze(1)
					if self.debug_count < 1:
						print('w/o attention forcing + w/ teacher forcing')
						self.debug_count += 1
					# print('here')
					# print(tgt[:, idx+1])
					# print(symbols.view(-1))
					# input('...')
				else:
					tgt_chunk = self.embedder_dec(symbols)
					if self.debug_count < 1:
						print('w/o attention forcing + w/o teacher forcing')
						self.debug_count += 1
				# print('target query size: {}'.format(tgt_chunk.size()))
				# print(lengths)
		else:
			# init
			# print('here')
			tgt_chunk_ref = tgt_chunk
			tgt_chunk_hyp = tgt_chunk
			cell_value_ref = cell_value
			cell_value_hyp = cell_value
			prev_c_ref = prev_c
			prev_c_hyp = prev_c
			dec_hidden_ref = dec_hidden
			dec_hidden_hyp = dec_hidden

			# tf at sentence level
			use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
			# loop
			for idx in range(self.max_seq_len - 1):

				assert type(att_scores) != type(None), 'empty att ref scores!'
				if isinstance(att_scores, list):
					# used in dual training
					step_attn_ref_detach = att_scores[idx].detach() 
				else:
					# used in fixed ref att training
					step_attn_ref_detach = att_scores[:, idx,:].unsqueeze(1)
				step_attn_ref_detach = step_attn_ref_detach.type(torch.FloatTensor).to(device=device)
				# print(step_attn_ref_detach.size())
				# print('here')
				# input('...')
				if self.debug_count < 1:
					print('w/ attention forcing')

				# hyp
				ret_dict[KEY_ATTN_REF].append(step_attn_ref_detach)
				predicted_softmax_hyp, dec_hidden_hyp, step_attn_hyp, c_out_hyp, cell_value_hyp = \
					self.forward_step(att_keys, att_vals, tgt_chunk_hyp, cell_value_hyp,
										dec_hidden_hyp, mask_src, prev_c_hyp, att_ref=step_attn_ref_detach)
				step_output_hyp = predicted_softmax_hyp.squeeze(1)
				symbols_hyp = decode(idx, step_output_hyp, step_attn_hyp)
				prev_c_hyp = c_out_hyp

				if idx < nb_tf_tokens: # use_teacher_forcing
					tgt_chunk_hyp = emb_tgt[:, idx+1].unsqueeze(1)
					# print('here')
					# print(tgt_chunk)
					if self.debug_count < 1:
						print('w/ teacher forcing')
						self.debug_count += 1

				else:
					tgt_chunk_hyp = self.embedder_dec(symbols_hyp)
					if self.debug_count < 1:
						print('w/o teacher forcing')
						self.debug_count += 1

				# tgt_chunk_hyp = self.embedder_dec(symbols_hyp)
				# print('step_attn_hyp', step_attn_hyp)
				# print(step_attn_ref_detach)
				# print(step_attn_hyp)
				# print(src)
				# print(tgt)
				# input('...')

		# print('...')
		ret_dict[KEY_SEQUENCE] = sequence_symbols
		ret_dict[KEY_LENGTH] = lengths.tolist()

		if hasattr(self, 'flag_stack_outputs'):
			if self.flag_stack_outputs:
				# print('stack')
				decoder_outputs = torch.stack(decoder_outputs, dim=2)
				ret_dict['attention_score'] = torch.cat(ret_dict['attention_score'], dim=1)
				ret_dict['attention_ref'] = torch.cat(ret_dict['attention_ref'], dim=1)

		return decoder_outputs, dec_hidden, ret_dict


