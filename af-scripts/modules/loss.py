from __future__ import print_function
import math
import torch 
import torch.nn as nn
import numpy as np

""" 
	from https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/loss/loss.py 
	with minor modification by YTL
"""

class Loss(object):
	""" Base class for encapsulation of the loss functions.
	This class defines interfaces that are commonly used with loss functions
	in training and inferencing.  For information regarding individual loss
	functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions
	Note:
		Do not use this class directly, use one of the sub classes.
	Args:
		name (str): name of the loss function used by logging messages.
		criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
			to http://pytorch.org/docs/master/nn.html#loss-functions for
			a list of them.
	Attributes:
		name (str): name of the loss function used by logging messages.
		criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
			to http://pytorch.org/docs/master/nn.html#loss-functions for
			a list of them.  Implementation depends on individual
			sub-classes.
		acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
		norm_term (float): normalization term that can be used to calculate
			the loss of multiple batches.  Implementation depends on individual
			sub-classes.
	"""

	def __init__(self, name, criterion):
		self.name = name
		self.criterion = criterion
		if not issubclass(type(self.criterion), nn.modules.loss._Loss):
			raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
		# accumulated loss
		self.acc_loss = 0
		# normalization term
		self.norm_term = 0

	def reset(self):
		""" Reset the accumulated loss. """
		self.acc_loss = 0
		self.norm_term = 0

	def get_loss(self):
		""" Get the loss.
		This method defines how to calculate the averaged loss given the
		accumulated loss and the normalization term.  Override to define your
		own logic.
		Returns:
			loss (float): value of the loss.
		"""
		raise NotImplementedError

	def eval_batch(self, outputs, target):
		""" Evaluate and accumulate loss given outputs and expected results.
		This method is called after each batch with the batch outputs and
		the target (expected) results.  The loss and normalization term are
		accumulated in this method.  Override it to define your own accumulation
		method.
		Args:
			outputs (torch.Tensor): outputs of a batch.
			target (torch.Tensor): expected output of a batch.
		"""
		raise NotImplementedError

	def cuda(self):
		self.criterion.cuda()

	def backward(self, retain_graph=False):
		if type(self.acc_loss) is int:
			raise ValueError("No loss to back propagate.")
		self.acc_loss.backward(retain_graph=retain_graph)

	def add(self, loss):
		if not issubclass(type(loss), Loss):
			raise ValueError("loss has to be a subclass of Loss")
		total_loss = self.acc_loss + loss.acc_loss
		return total_loss

	def mul(self, coeff):
		self.acc_loss *= coeff


class NLLLoss(Loss):
	""" Batch averaged negative log-likelihood loss.
	Args:
		weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
		mask (int, optional): index of masked token, i.e. weight[mask] = 0.
		size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
	"""
	"""
	YTL:
		set reduction = False to allow masking for each batch
	"""

	_NAME = "NLLLoss"

	def __init__(self, weight=None, mask=None, reduction='none'):
		self.mask = mask
		if mask is not None:
			if weight is None:
				raise ValueError("Must provide weight with a mask.")
			weight[mask] = 0

		# set loss as nn.NLLLoss
		super(NLLLoss, self).__init__(
			self._NAME,
			nn.NLLLoss(weight=weight, reduction=reduction))

	def get_loss(self):
		if isinstance(self.acc_loss, int):
			return 0
		# total loss for all batches
		loss = self.acc_loss.data.detach().item()
		loss /= self.norm_term
		return loss

	def eval_batch(self, outputs, target):
		self.acc_loss += torch.sum(self.criterion(outputs, target))
		self.norm_term += 1

	def eval_batch_with_mask(self, outputs, target, mask):
		# masked_loss = torch.mul(self.criterion(outputs, target),mask)
		masked_loss = self.criterion(outputs, target).masked_select(mask)
		self.acc_loss += masked_loss.sum()
		self.norm_term += 1

	def eval_batch_seq(self, outputs, target, outputs_fr=None, mask_utt_fr=None):
		"""
		outputs, target [B, D, T] = [batch, vocab size, Tout]; requirement of nn.NLLLoss
		"""
		loss = self.criterion(outputs, target)
		loss_utt = (loss).sum(1)
		if (outputs_fr is None) and (mask_utt_fr is None):
			# TF
			self.acc_loss += torch.sum(loss_utt)
		else:
			# AAF, i.e. TF+FR
			loss_fr = self.criterion(outputs_fr, target)
			loss_utt_fr = (loss_fr).sum(1)
			self.acc_loss += ((1-mask_utt_fr) * loss_utt).sum()
			self.acc_loss += (mask_utt_fr * loss_utt_fr).sum()
		self.norm_term += outputs.size()[2]

		# print('out sum', outputs.sum())
		# print(outputs[0,0,:3])
		# print('tgt sum', target.sum())
		# print(target[0,:3])
		# print('loss', loss_utt.sum())
		# print(loss_utt[:3])
		# if (mask_utt_fr is not None):
		# 	print('loss_fr', loss_utt_fr.sum())
		# 	print(loss_utt_fr[:3])


	def eval_batch_seq_with_mask(self, outputs, target, mask, outputs_fr=None, mask_utt_fr=None):
		
		# import pdb; pdb.set_trace()
		# print(outputs.size(), outputs_fr.size(), target.size(), mask.size())
		mask = mask.int().float()
		loss = self.criterion(outputs, target)
		loss_utt = (loss).sum(1) * mask
		if (outputs_fr is None) and (mask_utt_fr is None):
			# TF
			self.acc_loss += torch.sum(loss_utt)
		else:
			# AAF, i.e. TF+FR
			loss_fr = self.criterion(outputs_fr, target)
			loss_utt_fr = (loss_fr).sum(1) * mask

			self.acc_loss += ((1-mask_utt_fr) * loss_utt).sum()
			self.acc_loss += (mask_utt_fr * loss_utt_fr).sum()
		self.norm_term += outputs.size()[2]


class KLDivLoss(Loss):

	_NAME = "KLDivLoss"

	def __init__(self, reduction='none', fr_loss_max_rate=1.0, ep_aaf_start=10):

		# set loss as kl divergence
		super(KLDivLoss, self).__init__(
			self._NAME,
			nn.KLDivLoss(reduction=reduction))

		# for aaf
		self.mask_utt_fr = None
		self.fr_loss_max_rate = fr_loss_max_rate
		self.ep_aaf_start = ep_aaf_start
		self.fr_percent = -1.0

	def get_loss(self):
		if isinstance(self.acc_loss, int):
			return 0
		# total loss for all batches
		loss = self.acc_loss.data.detach().item()
		loss /= self.norm_term
		return loss

	def get_fr_percent(self):
		if isinstance(self.fr_percent, float):
			return self.fr_percent
		return self.fr_percent.detach().item()

	def eval_batch(self, outputs, target):

		# OPT 1 
		# # mask out zeros
		# mask1 = target.gt(0).detach()
		# mask2 = outputs.gt(0).detach()
		# mask3 = mask1 * mask2
		# # print(mask3)
		# # input('...')	

		# # compute loss
		# loss = self.criterion(torch.log(outputs), target)
		# masked_loss = loss * mask3.float()
		# # print(masked_loss)
		# # input('...')

		# self.acc_loss += masked_loss.sum()
		# self.norm_term += 1

		# OPT 2
		# remove zeros
		mask1 = target.gt(0.0)
		# print(mask1)
		target = target.masked_select(mask1)
		outputs = outputs.masked_select(mask1)
		# print(torch.log(outputs))
		# print(target)
		mask2 = outputs.gt(0.0)
		# print(mask2)
		# input('...')		
		target = target.masked_select(mask2)
		outputs = outputs.masked_select(mask2)
		# print(torch.log(outputs))
		# print(target)	

		# compute loss
		masked_loss = self.criterion(torch.log(outputs), target)
		self.acc_loss += masked_loss.sum()
		self.norm_term += 1
		print(masked_loss.sum())
		input('...')

	def eval_batch_with_mask(self, outputs, target, mask):

		# too slow
		# mask out zeros
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = mask1 * mask2
		# print('non zero mask', mask3)
		# input('...')	

		losses = []
		for idx in range(target.size(0)):
			ref = target[idx,:].masked_select(mask3[idx,:])
			hyp = outputs[idx,:].masked_select(mask3[idx,:])
			loss = self.criterion(torch.log(hyp), ref).sum()
			losses.append(loss)

		# print(losses)
		masked_loss = torch.stack(losses).masked_select(mask)
		# print('masked loss', masked_loss)
		# input('...')

		self.acc_loss += masked_loss.sum()		
		self.norm_term += 1


	def eval_batch_with_mask_v2(self, outputs, target, mask):
		# print(target.size(),target[0,:,:3])

		# mask out zeros
		mask0 = mask.view(-1,1).repeat(1, target.size(-1)).unsqueeze(1)
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = mask0 * mask1 * mask2 
		# print('mask', mask)
		# print('mask0', mask0.size())
		# print(mask0)
		# print('mask1', mask1.size())
		# print(mask1)
		# print('mask2', mask2.size())
		# print(mask2)
		# print('mask3', mask3.size())
		# print(mask3)
		# input('...')	

		target = target.masked_select(mask3)
		outputs = outputs.masked_select(mask3)
		masked_loss = self.criterion(torch.log(outputs), target)
		# print('masked loss', masked_loss.sum())
		# input('...')
		# print(mask3)

		self.acc_loss += masked_loss.sum()		
		self.norm_term += 1

	def eval_batch_seq_with_mask(self, outputs, target, mask, outputs_fr=None, epoch=None):
		"""
		unlike previous version, compute all steps in one go
		outputs, outputs_fr, target [B T D] = [B T_out-1 T_in]
		mask [B T] = [B T_out-1]
		"""

		# asup debug only
		# import pdb; pdb.set_trace()
		# print(outputs.size(), outputs_fr.size(), target.size(), mask.size())
		# mask0 = mask.unsqueeze(2).repeat(1, 1, target.size(-1))
		# mask1 = target.gt(0)
		# mask2 = outputs.gt(0)
		# mask3 = mask0 * mask1 * mask2
		# target = target.masked_select(mask3)
		# outputs = outputs.masked_select(mask3)
		# masked_loss = self.criterion(torch.log(outputs), target)
		# self.acc_loss += (masked_loss).sum()
		# self.norm_term += mask2.size()[1] # only norm over time for consistency


		# get mask_step [B T]; already there: mask
		# get loss [B T], average over time N get loss_utt [B]		
		mask0 = mask.unsqueeze(2).repeat(1, 1, target.size(-1))
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = (mask0 * mask1 * mask2).int().float().detach()

		eps = float(1e-10)
		outputs[outputs==0] = eps
		loss = (mask3 * self.criterion(torch.log(outputs), target)).sum(2)
		loss_utt = (loss).sum(1) # loss & loss*mask returns diff results, when using binary mask, but float mask is fine

		if (outputs_fr is None) and (epoch is None):
			self.acc_loss += (loss_utt).sum()
		else:
			mask2 = outputs_fr.gt(0)
			mask3 = (mask0 * mask1 * mask2).int().float().detach()
			outputs_fr[outputs_fr==0] = eps
			loss_fr = (mask3 * self.criterion(torch.log(outputs_fr), target)).sum(2)
			loss_utt_fr = (loss_fr).sum(1)

			# get mask_utt [B]
			self.mask_utt_fr = (loss_utt_fr < (loss_utt * self.fr_loss_max_rate)).int().float().detach()

			# get total loss
			self.acc_loss += ((1-self.mask_utt_fr) * loss_utt).sum()
			self.acc_loss += (self.mask_utt_fr * loss_utt_fr).sum()

			
		if torch.isnan(self.acc_loss).any():
			print('loss_utt', loss_utt)
			if outputs_fr is not None: print('loss_utt_fr', loss_utt_fr)
			import pdb; pdb.set_trace()
		assert not torch.isnan(self.acc_loss).any(), 'self.acc_loss is NaN, look into it' # tmp safety code
		self.norm_term += outputs.size()[1] # only norm over time for consistency

		# print('kl out', outputs[0,0,:3])
		# print('kl tgt', target[0,0,:3])
		# print('klloss', loss_utt.sum())
		if epoch:
			# print('klloss_fr', loss_utt_fr.sum())
			# print('fr rate', self.mask_utt_fr.mean())
			# print(float(self.mask_utt_fr.mean()),)
			self.fr_percent = self.mask_utt_fr.mean()

	def eval_batch_seq_with_mask_smooth(self, outputs, target, mask, outputs_fr=None, epoch=None):
		"""
		unlike previous version, compute all steps in one go
		outputs, outputs_fr, target [B T D] = [B T_out-1 T_in]
		mask [B T] = [B T_out-1]
		"""

		# asup debug only
		# import pdb;
		# print(outputs.size(), target.size(), mask.size())
		# print('out', outputs[0,0,:])
		# print('tgt', target[0,0,:])
		# pdb.set_trace()
	
		mask0 = mask.unsqueeze(2).repeat(1, 1, target.size(-1))
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = (mask0 * mask1 * mask2).int().float().detach()

		def smooth(d, eps = float(1e-10)):
			u = 1.0 / float(d.size()[2])
			return eps * u + (1-eps) * d

		# outputs += eps
		# target += eps
		loss = (mask3 * self.criterion(torch.log(smooth(outputs)), smooth(target))).sum(2)
		loss_utt = (loss).sum(1) # loss & loss*mask returns diff results, when using binary mask, but float mask is fine

		if (outputs_fr is None) and (epoch is None):
			self.acc_loss += (loss_utt).sum()
		else:
			mask2 = outputs_fr.gt(0)
			mask3 = (mask0 * mask1 * mask2).int().float().detach()
			outputs_fr[outputs_fr==0] = eps
			loss_fr = (mask3 * self.criterion(torch.log(smooth(outputs_fr)), smooth(target))).sum(2)
			loss_utt_fr = (loss_fr).sum(1)

			# get mask_utt [B]
			self.mask_utt_fr = (loss_utt_fr < (loss_utt * self.fr_loss_max_rate)).int().float().detach()

			# get total loss
			self.acc_loss += ((1-self.mask_utt_fr) * loss_utt).sum()
			self.acc_loss += (self.mask_utt_fr * loss_utt_fr).sum()

			
		if torch.isnan(self.acc_loss).any():
			print('loss_utt', loss_utt)
			if outputs_fr is not None: print('loss_utt_fr', loss_utt_fr)
			import pdb; pdb.set_trace()
		assert not torch.isnan(self.acc_loss).any(), 'self.acc_loss is NaN, look into it' # tmp safety code
		self.norm_term += outputs.size()[1] # only norm over time for consistency

		if epoch:
			self.fr_percent = self.mask_utt_fr.mean()

	def eval_batch_seq_with_mask_smooth_aoaf(self, outputs, target, mask, tf_ratio=None):
		"""
		unlike previous version, compute all steps in one go
		outputs, outputs_fr, target [B T D] = [B T_out-1 T_in]
		mask [B T] = [B T_out-1]
		"""

		# asup debug only
		# import pdb
	
		mask0 = mask.unsqueeze(2).repeat(1, 1, target.size(-1))
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = (mask0 * mask1 * mask2).int().float().detach()

		def smooth(d, eps = float(1e-10)):
			u = 1.0 / float(d.size()[2])
			return eps * u + (1-eps) * d

		loss = (mask3 * self.criterion(torch.log(smooth(outputs)), smooth(target))).sum(2)
		loss_utt = (loss).sum(1) # loss & loss*mask returns diff results, when using binary mask, but float mask is fine

		if (tf_ratio is None):
			self.acc_loss += (loss_utt).sum()
		else:
			# get mask_utt [B]
			l_utt = mask.float().sum(1)
			loss_utt_norm = loss_utt / l_utt
			tf_nb = int(loss_utt.size()[0] * tf_ratio)
			kl_max = torch.topk(loss_utt_norm, tf_nb)[0][-1] # use FR, if kl < kl_max
			self.mask_utt_fr = (loss_utt_norm < kl_max).int().float().detach()

			# get total loss
			self.acc_loss += (self.mask_utt_fr * loss_utt).sum()
			
		if torch.isnan(self.acc_loss).any():
			print('loss_utt', loss_utt)
			if outputs_fr is not None: print('loss_utt_fr', loss_utt_fr)
			import pdb; pdb.set_trace()
		assert not torch.isnan(self.acc_loss).any(), 'self.acc_loss is NaN, look into it' # tmp safety code
		self.norm_term += outputs.size()[1] # only norm over time for consistency

		# print('loss_utt', loss_utt)
		# print('loss_utt_norm', loss_utt_norm)
		# print('kl_max', kl_max)
		# print('self.mask_utt_fr', self.mask_utt_fr)
		# pdb.set_trace()


class MSELoss(Loss):

	_NAME = "MSELoss"

	def __init__(self, reduction='none', fr_loss_max_rate=1.0, ep_aaf_start=10):

		# set loss as kl divergence
		super(MSELoss, self).__init__(
			self._NAME,
			nn.MSELoss(reduction=reduction))

		# for aaf
		# self.mask_utt_fr = None
		# self.fr_loss_max_rate = fr_loss_max_rate
		# self.ep_aaf_start = ep_aaf_start
		# self.fr_percent = -1.0

	def get_loss(self):
		if isinstance(self.acc_loss, int):
			return 0
		# total loss for all batches
		loss = self.acc_loss.data.detach().item()
		loss /= self.norm_term
		return loss

	def eval_batch_seq_with_mask(self, outputs, target, mask):
		"""
		unlike previous version, compute all steps in one go
		outputs, outputs_fr, target [B T D] = [B T_out-1 T_in]
		mask [B T] = [B T_out-1]
		"""

		# asup debug only
		# import pdb; 
		# print(outputs.size(), target.size(), mask.size())
		# print('out', outputs[0,0,:])
		# print('tgt', target[0,0,:])
		# pdb.set_trace()
	
		mask0 = mask.unsqueeze(2).repeat(1, 1, target.size(-1))
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = (mask0 * mask1 * mask2).int().float().detach()

		# eps = float(1e-12)
		# outputs += eps
		# target += eps
		loss = (mask3 * self.criterion(outputs, target)).sum(2)
		loss_utt = (loss).sum(1) # loss & loss*mask returns diff results, when using binary mask, but float mask is fine

		# if (outputs_fr is None) and (epoch is None):
		self.acc_loss += (loss_utt).sum()
			
		if torch.isnan(self.acc_loss).any():
			print('loss_utt', loss_utt)
			import pdb; pdb.set_trace()
		assert not torch.isnan(self.acc_loss).any(), 'self.acc_loss is NaN, look into it' # tmp safety code
		self.norm_term += outputs.size()[1] # only norm over time for consistency


class NLLLoss_sched(NLLLoss):
	"""docstring for NLLLoss_sched"""
	def __init__(self, weight=None, mask=None, reduction='none'):
		super(NLLLoss_sched, self).__init__(weight=weight, mask=mask, reduction=reduction)

	def eval_batch_with_mask_sched(self, outputs, outputs_fr, target, mask, mask_utt_fr):
		
		# masked_loss = torch.mul(self.criterion(outputs, target),mask)
		masked_loss = self.criterion(outputs, target).masked_select(mask)
		masked_loss_fr = self.criterion(outputs_fr, target).masked_select(mask)

		self.acc_loss += ((~mask_utt_fr) * masked_loss).sum()
		self.acc_loss += (mask_utt_fr * masked_loss_fr).sum()		
		self.norm_term += 1

	def eval_batch_with_mask_v3(self, outputs, outputs_fr, target, mask, mask_utt_fr):
		
		# import pdb; pdb.set_trace()
		# print(outputs.size(), outputs_fr.size(), target.size(), mask.size())
		
		loss = self.criterion(outputs, target)
		loss_utt = (loss * mask).mean(1)
		loss_fr = self.criterion(outputs_fr, target)
		loss_utt_fr = (loss_fr * mask).mean(1)

		self.acc_loss += ((~mask_utt_fr) * loss_utt).sum()
		self.acc_loss += (mask_utt_fr * loss_utt_fr).sum()		
		self.norm_term += outputs.size()[2]
		

class KLDivLoss_sched(KLDivLoss):
	"""docstring for KLDivLoss_sched"""

	_NAME = "KLDivLoss_sched"

	def __init__(self, reduction='none'):
		super(KLDivLoss_sched, self).__init__(reduction=reduction)
		self.mask_utt_fr = None
		self.fr_loss_max_rate = 2.0

	def eval_batch_with_mask_v2_sched(self, outputs, outputs_fr, target, mask):

		# import pdb; pdb.set_trace()
		# print('outputs_fr', outputs_fr.size())
		# print('target', target.size())

		# mask out zeros
		mask0 = mask.view(-1,1).repeat(1, target.size(-1)).unsqueeze(1)
		mask1 = target.gt(0)
		mask2 = outputs.gt(0)
		mask3 = mask0 * mask1 * mask2 
		# print('mask', mask)
		# print('mask0', mask0.size())
		# print(mask0)
		# print('mask1', mask1.size())
		# print(mask1)
		# print('mask2', mask2.size())
		# print(mask2)
		# print('mask3', mask3.size())
		# print(mask3)
		# input('...')

		# import pdb; pdb.set_trace()
		
		target = target.masked_select(mask3)
		outputs = outputs.masked_select(mask3)
		masked_loss = self.criterion(torch.log(outputs), target)
		# print('masked loss', masked_loss)
		# input('...')


		# mask_fr = mask0 * mask1 * outputs_fr.gt(0)
		outputs_fr = outputs_fr.masked_select(mask3) # mask_fr
		masked_loss_fr = self.criterion(torch.log(outputs_fr), target)

		self.mask_utt_fr = masked_loss_fr < (masked_loss * self.fr_loss_max_rate)

		self.acc_loss += ((~self.mask_utt_fr) * masked_loss).sum()
		self.acc_loss += (self.mask_utt_fr * masked_loss_fr).sum()		
		self.norm_term += 1


	def eval_batch_with_mask_v3(self, outputs, outputs_fr, target, mask, epoch=None):
		"""
		unlike previous version, compute all steps in one go
		outputs, outputs_fr, target [B T D] = [B T_out-1 T_in]
		mask [B T+1] = [B T_out]
		"""

		# import pdb; pdb.set_trace()
		# print(outputs.size(), outputs_fr.size(), target.size(), mask.size())
		# print(outputs[0][0])
		# print(outputs[0][-1])
		# print(target[0][-1])
		# print(target[0][-1])

		# get mask_step [B T]; already there: mask

		# get loss [B T], average over time N get loss_utt [B]

		# mask = mask.int().float() # tmp fix
		eps = float(1e-10)
		outputs[outputs==0] = eps
		outputs_fr[outputs_fr==0] = eps


		loss = self.criterion(torch.log(outputs), target).mean(2)
		loss_utt = (loss * mask).mean(1)

		loss_fr = self.criterion(torch.log(outputs_fr), target).mean(2)
		loss_utt_fr = (loss_fr * mask).mean(1)

		# get mask_utt [B]
		self.mask_utt_fr = (epoch>10) * (loss_utt_fr < (loss_utt * self.fr_loss_max_rate) )

		# get total loss
		self.acc_loss += ((~self.mask_utt_fr) * loss_utt).sum()
		self.acc_loss += (self.mask_utt_fr * loss_utt_fr).sum()		

		if torch.isnan(self.acc_loss).any():
			import pdb; pdb.set_trace()
		assert not torch.isnan(self.acc_loss).any(), 'self.acc_loss is NaN, look into it' # tmp safety code
		self.norm_term += outputs.size()[1] # only norm over time for consistency
		



