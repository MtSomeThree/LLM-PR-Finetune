from NADO_ind import NADOInd
from constraints import LogicalConstraintFunction
from neural_constr import NeuralConstraintFunction
from sampling import load_prompts, sample_from_GPT2
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.optim import AdamW, Adam
from torch.nn.functional import log_softmax, softmax
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Sigmoid, LogSigmoid
import datasets
import copy
import torch
import argparse
import os
import time
import math

def train_nado(model, args):
	print ("Loading samples from %s"%(args.samples_file))
	samples_list, labels_list, logprobs_list, ptox_list, lengths_list = torch.load(args.samples_file, map_location=args.device)
	print ("Strat NADO Training...")
	for p in model.parameters():
		p.requires_grad = False
	nado_parameters = []
	for n, p in model.named_parameters():
		if "NADO" in n:
			p.requires_grad = True
			nado_parameters.append(p)

	print ("%d parameters in total"%(sum(p.numel() for p in model.parameters())))
	print ("%d parameters in nado"%(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	optimizer = AdamW(params=nado_parameters, lr=args.lr)

	for epoch in range(args.num_epochs):
		cnt = 1
		loss_list = []
		for samples, labels, logprobs, ptox, length in zip(samples_list, labels_list, logprobs_list, ptox_list, lengths_list):
			if length > 30:
				continue
			labels = labels.float()
			#probs = softmax(logprobs, dim=0) * float(labels.shape[0])
			masks = torch.where(samples == model.config.eos_token_id, 0, 1)
			for i in range(labels.shape[0]):
				if labels[i] == -1.0:
					masks[i] *= 0
				else:
					if args.threshold is not None:
						labels[i] = 0.0 if labels[i] < args.threshold else 1.0
					else:
						labels[i] = math.pow(labels[i], args.temperature)
			outputs = model(input_ids=samples, attention_mask=masks, rc_weights=logprobs, rc_labels=1.0 - labels, prompt_length=length)
			loss = outputs.loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			cnt += 1
			loss_list.append(loss.item())

			if cnt % 1000 == 0:
				print ("%d processed: avg loss: %.4f"%(cnt, torch.Tensor(loss_list).mean()))

		print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

		#satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=True, sample_text=(epoch == args.num_epochs - 1))
		#print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))

	model.save_NADO_to_cache(args.save_dir)

def PR_finetune(model, teacher_model, args):
	samples_list, labels_list, logprobs_list, ptox_list, lengths_list = torch.load(args.samples_file, map_location=args.device)
	optimizer = AdamW(params=model.parameters(), lr=args.lr)
	loss_fct = CrossEntropyLoss()

	for epoch in range(args.num_epochs):
		cnt = 1
		loss_list = []
		for group_samples, group_labels, group_logprobs, ptox, length in zip(samples_list, labels_list, logprobs_list, ptox_list, lengths_list):
			if length > 30:
				continue
			num_batches = int((group_samples.shape[0] - 1) / args.batch_size) + 1
			for idx in range(num_batches):
				samples = group_samples[idx * args.batch_size: (idx + 1) * args.batch_size]
				labels = group_labels[idx * args.batch_size: (idx + 1) * args.batch_size]
				logprobs = group_logprobs[idx * args.batch_size: (idx + 1) * args.batch_size]
				
				labels = labels.float()
				#probs = softmax(logprobs, dim=0) * float(labels.shape[0])
				masks = torch.where(samples == model.config.eos_token_id, 0, 1)
				masks = masks.to(args.device)
				nado_outputs = teacher_model(input_ids=samples, attention_mask=masks, rc_weights=logprobs)
				truth_logits = nado_outputs.logits[:, length:, :]
				truth_probs = softmax(truth_logits, dim=2)
				outputs = model(input_ids=samples, attention_mask=masks, return_dict=True)
				student_logits = outputs.logits[:, length:, :]
				
				loss = loss_fct(student_logits.permute(0, 2, 1), truth_probs.permute(0, 2, 1))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				loss_list.append(loss.item())
			cnt += 1
			if cnt % 1000 == 0:
				print ("%d processed: avg loss: %.4f"%(cnt, torch.Tensor(loss_list).mean()))

		print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

	torch.save(model.state_dict(), args.save_dir)

def baseline_finetune(model, args):
	samples_list, labels_list, logprobs_list, ptox_list, lengths_list = torch.load(args.samples_file, map_location=args.device)
	finetune_parameters = []
	for n, p in model.named_parameters():
		if "nado" in n:
			p.requires_grad = False
		else:
			finetune_parameters.append(p)

	optimizer = Adam(params=finetune_parameters, lr=args.lr)
	loss_fct = CrossEntropyLoss()

	for epoch in range(args.num_epochs):
		cnt = 1
		loss_list = []
		for group_samples, group_labels, group_logprobs, ptox, length in zip(samples_list, labels_list, logprobs_list, ptox_list, lengths_list):
			if length > 30:
				continue
			num_batches = int((group_samples.shape[0] - 1) / args.batch_size) + 1
			for idx in range(num_batches):
				samples = group_samples[idx * args.batch_size: (idx + 1) * args.batch_size]
				labels = group_labels[idx * args.batch_size: (idx + 1) * args.batch_size]
				logprobs = group_logprobs[idx * args.batch_size: (idx + 1) * args.batch_size]
				
				labels = torch.tensor(labels.float())
				ids = labels.le(labels.median())
				pos_samples = samples[ids]
				pos_labels = labels[ids]

				#probs = softmax(logprobs, dim=0) * float(labels.shape[0])
				masks = torch.where(pos_samples == model.config.eos_token_id, 0, 1)
				masks = masks.to(args.device)
				outputs = model(input_ids=pos_samples, attention_mask=masks, labels=pos_samples)
				loss = outputs.loss

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				loss_list.append(loss.item())
			cnt += 1
			if cnt % 1000 == 0:
				print ("%d processed: avg loss: %.4f"%(cnt, torch.Tensor(loss_list).mean()))

		print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

	torch.save(model.state_dict(), args.save_dir)

def generate_for_eval(model, tokenizer, prompts, toxicity, args):
	args.samples_file = args.save_dir
	sample_from_GPT2(model, tokenizer, prompts, toxicity, args)

if __name__ == "__main__":
	time_start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--samples_per_prompt', type=int, default=200)
	parser.add_argument('--batch_size', type=int, default=25)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.00003)
	parser.add_argument('--cuda', type=int, default=-1)
	parser.add_argument('--num_test', type=int, default=100)
	parser.add_argument('--new_length', type=int, default=25)
	parser.add_argument('--max_prompt_length', type=int, default=30)
	parser.add_argument('--device', type=str, default=None)
	parser.add_argument('--save_dir', type=str, default=None)
	parser.add_argument('--load_dir', type=str, default=None)
	parser.add_argument('--load_dir2', type=str, default=None)
	parser.add_argument('--fine_tune', action='store_true')
	parser.add_argument('--eval_model', type=str, default=None)
	parser.add_argument('--constraint_id', type=int, default=1)
	parser.add_argument('--samples_file', type=str, default=None)
	parser.add_argument('--temperature', type=float, default=1.0)
	parser.add_argument('--threshold', type=float, default=None)
	parser.add_argument('--eval_start', type=int, default=10000)
	parser.add_argument('--eval_end', type=int, default=11000)
	parser.add_argument('--baseline', action='store_true')

	args = parser.parse_args()

	#constraint_function = LogicalConstraintFunction(args.constraint_id)
	#constraint_function = NeuralConstraintFunction()
	#constraint_function.init_formality()	

	if args.device is None:
		if args.cuda == -1:
			args.device = 'cpu'
		else:
			args.device = "cuda:%d"%(args.cuda)

	if args.samples_file is None:
		args.samples_file = './dump/toxicity_10000-200-50-API.pt'

	if args.samples_file is not None and 'dump' not in args.samples_file:
		args.samples_file = 'dump/' + args.samples_file
	if args.save_dir is not None and 'dump' not in args.save_dir:
		args.save_dir = 'dump/' + args.save_dir
	if args.load_dir is not None and 'dump' not in args.load_dir:
		args.load_dir = 'dump/' + args.load_dir	
	if args.load_dir2 is not None and 'dump' not in args.load_dir2:
		args.load_dir2 = 'dump/' + args.load_dir2	

	if args.save_dir is not None and args.load_dir is not None:
		print ("Loading from %s and saving to %s"%(args.load_dir, args.save_dir))

	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	# if args.load_dir is not None:
	# 	model.model_rc.load_state_dict(torch.load(args.load_dir))
	# 	model.to(args.device)
	# 	satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=False, sample_text=False)
	# 	print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))
	# else:
	if args.baseline:
		model = NADOInd.from_pretrained("gpt2")
		if args.load_dir is not None:
			model.load_state_dict(torch.load(args.load_dir, map_location=args.device), strict=False)
		model.to(args.device)
		model.set_constraint_factor(0.0)
		baseline_finetune(model, args)
	elif args.fine_tune:
		model = NADOInd.from_pretrained("gpt2")
		model.load_NADO_from_cache(args.load_dir, args.device)
		model.to(args.device)
		student_model = GPT2LMHeadModel.from_pretrained("gpt2")
		student_model.to(args.device)
		PR_finetune(student_model, model, args)
	elif args.eval_model is not None:
		prompts, toxicity = load_prompts()
		eval_prompts = prompts[args.eval_start: args.eval_end]
		eval_toxicity = toxicity[args.eval_start: args.eval_end]
		if args.eval_model == 'nado':
			model = NADOInd.from_pretrained("gpt2")
			model.load_NADO_from_cache(args.load_dir, args.device)
			
		elif args.eval_model == 'gpt':
			model = GPT2LMHeadModel.from_pretrained("gpt2")
			if args.load_dir is not None:
				model.load_state_dict(torch.load(args.load_dir, map_location=args.device))

		elif args.eval_model == 'llama':
			model = model = LlamaForCausalLM.from_pretrained("../llama/7B")
			tokenizer = LlamaTokenizer.from_pretrained("../llama/7B")
		else:
			model = NADOInd.from_pretrained("gpt2")
			model.load_state_dict(torch.load(args.load_dir, map_location=args.device), strict=False)
		model.to(args.device)
		generate_for_eval(model, tokenizer, eval_prompts, eval_toxicity, args)
	else:
		model = NADOInd.from_pretrained("gpt2")
		if args.load_dir is not None:
			model.load_state_dict(torch.load(args.load_dir, map_location=args.device), strict=False)
		if args.load_dir2 is not None:
			model.load_NADO_from_cache(args.load_dir2, device=args.device)
		model.to(args.device)
		#satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=False, sample_text=False)
		#print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))
		#samples_list, labels_list, masks_list, logprobs_list = sample_from_GPT2(model, tokenizer, constraint_function, args)
		model.set_constraint_factor(1.0)
		train_nado(model, args)

	time_end = time.time()
	print ("Running Time (h):", (time_end - time_start) / 3600)


	

