prefix=$1
#python perspectiveAPI.py --filename dump/GPT2_20000-30000-200-20-None.pt
#python perspectiveAPI.py --filename dump/GPT2_30000-40000-200-20-None.pt &
python train_nado.py --cuda 2 --num_epochs 5 --load_dir2 "$prefix"/NADO_0_5.pt --save_dir "$prefix"/NADO_1_5.pt  --samples_file dump/GPT2_20000-30000-200-20-API.pt
wait
for i in {2..6}
do
	let j=i-1
	let start=(i+1)*10000
	let end=(i+2)*10000
	let next=(i+3)*10000
	python train_nado.py --cuda 2 --num_epochs 5 --load_dir2 "$prefix"/NADO_"$j"_5.pt --save_dir "$prefix"/NADO_"$i"_5.pt  --samples_file dump/GPT2_"$start"-"$end"-200-20-API.pt
	python sampling.py --API --cuda 6 --prompts_num 10000 --start_index $start --model_name GPT2 --samples_file dump/"$prefix"/GPT2_"$end"-"$next"-200-20-None.pt
done