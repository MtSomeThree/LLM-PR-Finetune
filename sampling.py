from constraints import LogicalConstraintFunction
from neural_constr import NeuralConstraintFunction
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from torch.optim import AdamW, Adam
from torch.nn.functional import log_softmax, softmax
from perspectiveAPI import fix_samples
import pickle
import datasets
import copy
import torch
import argparse
import os
import json
import time

def sample_from_GPT2(model, tokenizer, prompts, toxicities, args):
	labels_list = []
	samples_list = []
	logprobs_list = []
	toxicity_list = []
	lengths_list = []
	cnt = 0
	for prompt, toxicity in zip(prompts, toxicities):

		num_bins = int(args.samples_per_prompt / args.batch_size)
		sequences = None
		logprobs = None

		for i in range(num_bins):

			sentence_prefix = [prompt] * args.batch_size
			encodings_dict = tokenizer.batch_encode_plus(sentence_prefix)

			input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
			attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)

			if input_ids.shape[1] > args.max_prompt_length:
				input_ids = input_ids[:, :args.max_prompt_length]
				attention_mask = attention_mask[:, :args.max_prompt_length]

			outputs = model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				do_sample=True,
				max_new_tokens=args.new_length,  # desired output sentence length
				pad_token_id=model.config.eos_token_id,
				output_scores=True,
				return_dict_in_generate=True,
			)

			transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
			output_logprobs = transition_scores.sum(axis=1)

			if sequences is None:
				sequences = outputs.sequences.cpu()
			else:
				sequences = torch.vstack((sequences, outputs.sequences.cpu()))

			if logprobs is None:
				logprobs = output_logprobs.cpu()
			else:
				logprobs = torch.vstack((logprobs, output_logprobs.cpu()))

		labels = torch.Tensor([-1] * args.samples_per_prompt)
		labels_list.append(labels)
		samples_list.append(outputs.sequences)
		logprobs_list.append(logprobs)
		toxicity_list.append(toxicity)
		lengths_list.append(input_ids.shape[1])
		cnt += 1
		if cnt % 500 == 0:
			print ("%d prompts are processed..."%(cnt))
		

	torch.save((samples_list, labels_list, logprobs_list, toxicity_list, lengths_list), args.samples_file)
	return (samples_list, labels_list, logprobs_list, toxicity_list, lengths_list)

def load_prompts(filename=None, shuffled=True):
	if filename is None:
		filename = './rtp/prompts.jsonl'
	if shuffled:
		prompts, toxicity = pickle.load(open('./rtp/prompts_shuffled.pkl', 'rb'))
	else:
		prompts = []
		toxicity = []
		with open(filename, 'r') as f:
			jlist = list(f)
			for jstr in jlist:
				result = json.loads(jstr)
				prompts.append(result['prompt']['text'])
				toxicity.append(result['prompt']['toxicity'])
	return prompts, toxicity


def fine_tune_GPT2_with_pos_samples(model, samples_list, labels_list, masks_list, logprobs_list, args):
	model.set_constraint_factor(0.0)

	fine_tune_parameters = []
	for n, p in model.named_parameters():
		if "model_rc" in n:
			p.requires_grad = False
		else:
			fine_tune_parameters.append(p)

	optimizer = Adam(params=fine_tune_parameters, lr=args.lr)


	for epoch in range(args.num_epochs):
		cnt = 1
		loss_list = []
		for samples, labels, masks, logprobs in zip(samples_list, labels_list, masks_list, logprobs_list):
			labels = labels.float()
			if labels.sum() < 0.5:
				continue
			outputs = model(input_ids=samples, attention_mask=masks, labels=samples, rc_weights=logprobs, rc_labels=labels.squeeze(1))
			loss = outputs.loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			cnt += 1
			loss_list.append(loss.item())

		print ("Epoch %d: avg loss: %.4f"%(epoch, torch.Tensor(loss_list).mean()))

		satisfied = test_rc(model, tokenizer, constraint_function, args, use_constr=True, sample_text=(epoch == args.num_epochs - 1))
		print (float(satisfied) / float(args.num_test) / float(args.sample_batch_size))


if __name__ == "__main__":
	time_start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--samples_per_prompt', type=int, default=200)
	parser.add_argument('--prompts_num', type=int, default=10000)
	parser.add_argument('--batch_size', type=int, default=50)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.00003)
	parser.add_argument('--cuda', type=int, default=-1)
	parser.add_argument('--num_test', type=int, default=100)
	parser.add_argument('--new_length', type=int, default=20)
	parser.add_argument('--max_prompt_length', type=int, default=30)
	parser.add_argument('--device', type=str, default=None)
	parser.add_argument('--dump_dir', type=str, default=None)
	parser.add_argument('--load_dir', type=str, default=None)
	parser.add_argument('--samples_file', type=str, default=None)
	parser.add_argument('--start_index', type=int, default=0)
	parser.add_argument('--model_name', type=str, default='GPT2')
	parser.add_argument('--API', action='store_true')

	args = parser.parse_args()

	if args.device is None:
		if args.cuda == -1:
			args.device = 'cpu'
		else:
			args.device = "cuda:%d"%(args.cuda)

	if args.samples_file is None:
		args.samples_file = './dump/%s_%d-%d-%d-%d-None.pt'%(args.model_name, args.start_index, args.start_index + args.prompts_num, args.samples_per_prompt, args.new_length)

	model = GPT2LMHeadModel.from_pretrained("gpt2")
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	if args.model_name != 'GPT2' and args.load_dir is None:
		args.load_dir = args.model_name + '.pt'
	if args.load_dir is not None:
		if 'dump' not in args.load_dir:
			args.load_dir = 'dump/' + args.load_dir
		model.load_state_dict(torch.load(args.load_dir, map_location=args.device))
	model.to(args.device)
	prompts, toxicity = load_prompts('./rtp/prompts.jsonl')

	sample_from_GPT2(model, tokenizer, prompts[args.start_index: args.start_index + args.prompts_num], toxicity[args.start_index: args.start_index + args.prompts_num], args)
	if args.API:
		fix_samples(args.samples_file, tokenizer)
	time_end = time.time()
	print ("Running Time (h):", (time_end - time_start) / 3600)

