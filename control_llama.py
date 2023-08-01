from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("../llama/7B")
model = LlamaForCausalLM.from_pretrained("../llama/7B")