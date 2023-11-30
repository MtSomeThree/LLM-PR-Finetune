import argparse
import json
from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaModel
from datasets import load_dataset
import torch
import time

def oracle(texts, pipeline):
	outputs = pipeline(texts)
	score = 0.0
	for output in outputs:
		if output['label'] == 'LABEL_1':
			score += 1.0
	return score

def reinforcement_train(model, tokenizer, pipeline, train_data, test_data, args):
	cnt = 0
	total_score = 0.0
	for data in test_data:
		prompt = data['text']
		prompts = [prompt] * args.batch_size
		encodings_dict = tokenizer(prompts)
		input_ids = torch.tensor(encodings_dict['input_ids']).to(args.device)
		attention_mask = torch.tensor(encodings_dict['attention_mask']).to(args.device)
		prompt_length = len(input_ids[0])
		outputs = model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				do_sample=True,
				temperature=0.1,
				top_p=0.9,
				max_new_tokens=args.max_length,  # desired output sentence length
				pad_token_id=model.config.eos_token_id,
				output_scores=True,
				return_dict_in_generate=True,
			)
		gen_text = tokenizer.batch_decode(outputs['sequences'][:, prompt_length:])
		score = oracle(gen_text, pipeline)
		total_score += score
		cnt += 1
		if cnt >= args.size:
			break
	print (total_score / float(cnt) / float(args.batch_size))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda', type=int, default=-1)
	parser.add_argument('--cuda2', type=int, default=2)
	parser.add_argument('--size', type=int, default=100)
	parser.add_argument('--device', type=str, default=None)
	parser.add_argument('--oracle_device', type=str, default=None)
	parser.add_argument('--max_length', type=int, default=32)
	parser.add_argument('--batch_size', type=int, default=1)
	args = parser.parse_args()

	if args.cuda == -1:
		args.device = 'cpu'
	else:
		args.device = 'cuda:%d'%(args.cuda)
	if args.cuda2 == -1:
		args.oracle_device = 'cpu'
	else:
		args.oracle_device = 'cuda:%d'%(args.cuda2)
	tokenizer = LlamaTokenizer.from_pretrained("../llama/7B")
	model = LlamaForCausalLM.from_pretrained("../llama/7B").to(args.device)
	TG_data_train = load_dataset("skg/toxigen-data", name="train", use_auth_token='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') # 250k training examples
	TG_data_train = TG_data_train.shuffle(seed=42)
	TG_data_train = TG_data_train['train']
	f = open('SafeNLP/data/toxiGen.json')
	test_data = json.load(f)
	toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta", device=args.cuda2)
	start_time = time.time()
	reinforcement_train(model, tokenizer, toxigen_roberta, TG_data_train, test_data, args)
	end_time = time.time()

	print ("Total Time: ", end_time - start_time)
	print (toxigen_roberta("I like playing the piano."))